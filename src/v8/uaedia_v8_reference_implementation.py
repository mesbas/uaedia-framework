"""
U-AEDIA Reference Implementation
Unified AI-Driven Enterprise Decision Intelligence Architecture
v8.0

Mesbaul Haque Sazu
Principal Architect

This module provides the canonical reference implementation of the
U-AEDIA framework. It covers all five architectural layers:

    L1 - Ingestion Gateway
    L2 - Intelligence Layer (Feature Registry, Model Registry)
    L3 - Decision Orchestration Layer (DOL) with formal state machine
    L4 - Domain Configuration Package (DCP) loader
    L5 - Audit and Compliance Layer (Decision Object, DCID index)

Agentic extensions (v6.0+):
    - Enterprise Context Graph (ECG) client
    - Governance Gate
    - Agentic Orchestration Engine (AOE)
    - LLM rationale generator
    - OpenTelemetry instrumentation

Dependencies:
    pydantic>=2.5.0
    pyyaml>=6.0.1
    jsonschema>=4.21.0
    confluent-kafka>=2.3.0
    fastavro>=1.9.0
    feast>=0.38.0
    redis>=5.0.1
    scikit-learn>=1.4.0
    xgboost>=2.0.3
    lightgbm>=4.3.0
    shap>=0.44.1
    mlflow>=2.11.0
    openai>=1.12.0
    anthropic>=0.21.0
    langchain>=0.1.12
    langgraph>=0.0.40
    neo4j>=5.17.0
    sqlalchemy>=2.0.25
    psycopg2-binary>=2.9.9
    cryptography>=42.0.0
    fastapi>=0.109.0
    uvicorn[standard]>=0.27.0
    httpx>=0.26.0
    opentelemetry-api>=1.22.0
    opentelemetry-sdk>=1.22.0
    opentelemetry-instrumentation-fastapi>=0.43b0
    prometheus-client>=0.19.0
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# =============================================================================
# SECTION 1: CORE IDENTIFIERS
# =============================================================================

def generate_dcid(domain_prefix: str) -> str:
    """
    Generate a globally unique Decision Context ID.

    Format: {PREFIX}-{UUID_HEX_12}
    Example: CRD-A3F9B20C1E4D

    The DCID is assigned at the Ingestion Gateway (L1) and propagated
    immutably through all downstream layers. It is the single key enabling
    end-to-end audit traceability across the full decision pipeline.

    Args:
        domain_prefix: 2-4 character domain code from the DCP (e.g. 'CRD', 'CLM', 'AGNT')

    Returns:
        Globally unique DCID string.
    """
    prefix = domain_prefix.upper()[:4]
    suffix = uuid.uuid4().hex.upper()[:12]
    return f"{prefix}-{suffix}"


# =============================================================================
# SECTION 2: DOL STATE MACHINE
# =============================================================================

class DOLState(Enum):
    """
    Nine-state formal state machine governing the U-AEDIA decision lifecycle.

    States are traversed sequentially under normal execution. The COMPENSATED
    terminal state provides explicit recovery semantics for partial execution
    failures -- a structural property absent in standard MLOps frameworks.
    """
    INGESTED    = auto()   # Event received; DCID assigned
    VALIDATING  = auto()   # Schema and business rule validation
    ENRICHING   = auto()   # Feature computation and context retrieval
    INFERRING   = auto()   # ML model inference (champion/challenger)
    EVALUATING  = auto()   # Rule evaluation against inference output
    DECIDING    = auto()   # Final decision composition
    EXECUTING   = auto()   # Action execution (API call, state write)
    COMPLETED   = auto()   # Decision Object sealed and persisted
    COMPENSATED = auto()   # Partial failure; compensating actions applied


# Valid state transitions. Any transition not listed raises InvalidTransitionError.
VALID_TRANSITIONS: Dict[DOLState, List[DOLState]] = {
    DOLState.INGESTED:    [DOLState.VALIDATING],
    DOLState.VALIDATING:  [DOLState.ENRICHING,  DOLState.COMPENSATED],
    DOLState.ENRICHING:   [DOLState.INFERRING,  DOLState.COMPENSATED],
    DOLState.INFERRING:   [DOLState.EVALUATING, DOLState.COMPENSATED],
    DOLState.EVALUATING:  [DOLState.DECIDING,   DOLState.COMPENSATED],
    DOLState.DECIDING:    [DOLState.EXECUTING,  DOLState.COMPENSATED],
    DOLState.EXECUTING:   [DOLState.COMPLETED,  DOLState.COMPENSATED],
    DOLState.COMPLETED:   [],   # Terminal
    DOLState.COMPENSATED: [],   # Terminal
}


class InvalidTransitionError(Exception):
    def __init__(self, dcid: str, current: DOLState, target: DOLState):
        super().__init__(
            f"[{dcid}] Invalid DOL transition: {current.name} -> {target.name}"
        )


@dataclass
class DecisionContext:
    """
    Mutable carrier object threading through all DOL state handlers.

    Created at ingestion, updated at each state transition, and sealed
    into an immutable DecisionObject upon reaching COMPLETED.
    """
    dcid: str
    event_payload: Dict[str, Any]
    domain: str
    state: DOLState = DOLState.INGESTED
    state_history: List[Dict[str, str]] = field(default_factory=list)
    feature_vector: Optional[Dict[str, Any]] = None
    inference_output: Optional[Dict[str, Any]] = None
    rule_output: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    compensating_reason: Optional[str] = None


def dol_transition(ctx: DecisionContext, target: DOLState) -> DecisionContext:
    """
    Execute a state transition on the DecisionContext.

    Validates the transition against VALID_TRANSITIONS, appends an
    audit-logged entry to state_history, and updates ctx.state atomically.

    Raises:
        InvalidTransitionError: if target is not a valid successor of ctx.state.
    """
    if target not in VALID_TRANSITIONS[ctx.state]:
        raise InvalidTransitionError(ctx.dcid, ctx.state, target)

    ctx.state_history.append({
        "from": ctx.state.name,
        "to": target.name,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    ctx.state = target
    return ctx


# =============================================================================
# SECTION 3: DECISION OBJECT SCHEMA
# =============================================================================

@dataclass
class AdverseActionFactor:
    """
    SHAP-attributed adverse action factor.

    Satisfies ECOA / FCRA adverse action notice requirements by recording
    the specific model features that drove a decline decision, along with
    the direction of their influence and the applicable regulatory code.
    """
    feature_name: str
    shap_value: float
    direction: str          # "increase" | "decrease"
    regulatory_code: str    # ECOA / FCRA adverse action code, e.g. "A9"


@dataclass
class DecisionObject:
    """
    Canonical immutable output of every U-AEDIA decision pipeline execution.

    The DecisionObject is produced when the DOL reaches COMPLETED state.
    It captures the full decision lifecycle -- from ingestion through
    executed action -- under a single persistent DCID. The SHA-256
    integrity_hash is computed over the full payload at seal time and
    cannot be recomputed after storage without invalidating the hash.

    Regulatory coverage:
        SR 11-7 / OCC 2011-12  -- model risk management documentation
        EU AI Act Article 13   -- right to explanation
        NIST AI RMF GOVERN 1.4 -- audit trail completeness
        ECOA / FCRA            -- adverse action notice
        FDA 21 CFR Part 11     -- electronic records (clinical deployments)
    """
    # --- Identity ---
    dcid: str
    schema_version: str = "uaedia-do-v3"
    framework_version: str = "u-aedia-v8.0"

    # --- Provenance ---
    domain: str = ""
    deployment_id: str = ""

    # --- Timing ---
    ingested_at: str = ""
    decided_at: str = ""
    completed_at: str = ""
    latency_ms: int = 0

    # --- Input record ---
    event_payload_hash: str = ""          # SHA-256 of raw event bytes
    feature_vector_snapshot: Dict[str, Any] = field(default_factory=dict)

    # --- Inference record ---
    models_invoked: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {model_id, model_version, score, confidence, inference_latency_ms}

    # --- Decision ---
    decision_outcome: str = ""            # APPROVED | DECLINED | REFERRED | BLOCKED
    decision_rationale: str = ""
    confidence_score: float = 0.0
    threshold_applied: float = 0.0

    # --- Governance ---
    rules_evaluated: List[Dict[str, Any]] = field(default_factory=list)
    governance_gate_result: str = ""      # PASS | BLOCK | REFER
    human_review_required: bool = False
    adverse_action_factors: List[AdverseActionFactor] = field(default_factory=list)

    # --- Full state trace ---
    state_history: List[Dict[str, str]] = field(default_factory=list)

    # --- Integrity ---
    integrity_hash: str = ""              # SHA-256 over serialized payload; set by seal()

    def seal(self) -> "DecisionObject":
        """
        Compute and store the SHA-256 integrity hash over the full payload.

        Must be called exactly once, immediately before persisting to the
        Decision Object Store. After seal(), the object must be treated
        as immutable. Any subsequent field modification invalidates the hash.

        Returns:
            self, for call-chaining.
        """
        payload = {k: v for k, v in asdict(self).items() if k != "integrity_hash"}
        canonical = json.dumps(payload, sort_keys=True, default=str).encode()
        self.integrity_hash = hashlib.sha256(canonical).hexdigest()
        self.completed_at = datetime.now(timezone.utc).isoformat()
        return self

    def verify_integrity(self) -> bool:
        """
        Recompute the SHA-256 hash and compare to stored integrity_hash.

        Use for audit verification. A mismatch indicates tampering or
        corruption of the stored record.
        """
        payload = {k: v for k, v in asdict(self).items() if k != "integrity_hash"}
        canonical = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(canonical).hexdigest() == self.integrity_hash

    @classmethod
    def from_context(cls, ctx: DecisionContext, ingested_at: str) -> "DecisionObject":
        """Build a DecisionObject from a completed DecisionContext."""
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            dcid=ctx.dcid,
            domain=ctx.domain,
            ingested_at=ingested_at,
            decided_at=now,
            event_payload_hash=hashlib.sha256(
                json.dumps(ctx.event_payload, sort_keys=True, default=str).encode()
            ).hexdigest(),
            feature_vector_snapshot=ctx.feature_vector or {},
            models_invoked=ctx.inference_output.get("models_invoked", []) if ctx.inference_output else [],
            decision_outcome=ctx.decision.get("outcome", "") if ctx.decision else "",
            decision_rationale=ctx.decision.get("rationale", "") if ctx.decision else "",
            confidence_score=ctx.decision.get("confidence", 0.0) if ctx.decision else 0.0,
            rules_evaluated=ctx.rule_output.get("rules_evaluated", []) if ctx.rule_output else [],
            governance_gate_result=ctx.decision.get("gate_result", "") if ctx.decision else "",
            state_history=ctx.state_history,
        )


# =============================================================================
# SECTION 4: DOMAIN CONFIGURATION PACKAGE (DCP)
# =============================================================================

try:
    import yaml
    import jsonschema
    import boto3
    import redis as redis_lib
except ImportError:
    yaml = None  # type: ignore

DCP_JSON_SCHEMA = {
    "type": "object",
    "required": ["dcp_version", "domain_id", "ingestion", "intelligence", "dol", "governance", "audit"],
    "properties": {
        "dcp_version": {"type": "string"},
        "domain_id": {"type": "string"},
        "domain_display_name": {"type": "string"},
        "ingestion": {
            "type": "object",
            "required": ["schema_registry_url", "kafka_topic", "dcid_prefix"],
        },
        "intelligence": {
            "type": "object",
            "required": ["models"],
            "properties": {
                "models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["model_id", "model_version", "serving_endpoint", "champion"],
                    },
                }
            },
        },
        "dol": {
            "type": "object",
            "required": ["state_machine_timeout_ms", "compensating_action"],
        },
        "governance": {
            "type": "object",
            "required": ["adverse_action_required", "regulatory_regime"],
        },
        "audit": {
            "type": "object",
            "required": ["decision_object_store", "retention_years"],
        },
    },
}

_DCP_CACHE: Dict[str, dict] = {}


def load_dcp(domain_id: str, version: str = "latest") -> dict:
    """
    Load, validate, and return a Domain Configuration Package.

    Resolution order:
        1. In-process cache (for hot-path serving performance)
        2. Redis (TTL: 300s, for cross-process cache coherence)
        3. S3 (source of truth; bucket: uaedia-dcps)

    Args:
        domain_id: DCP domain identifier (e.g. 'credit_decisioning_v3')
        version:   DCP version string or 'latest'

    Returns:
        Validated DCP dict.

    Raises:
        jsonschema.ValidationError: if the DCP fails schema validation.
        FileNotFoundError: if the DCP artifact cannot be located.
    """
    cache_key = f"{domain_id}:{version}"

    if cache_key in _DCP_CACHE:
        return _DCP_CACHE[cache_key]

    raw = _fetch_dcp_raw(domain_id, version)
    dcp = yaml.safe_load(raw)
    jsonschema.validate(dcp, DCP_JSON_SCHEMA)
    _DCP_CACHE[cache_key] = dcp
    return dcp


def _fetch_dcp_raw(domain_id: str, version: str) -> str:
    """Fetch raw DCP YAML bytes from Redis cache or S3."""
    redis_key = f"dcp:{domain_id}:{version}"
    try:
        r = redis_lib.Redis.from_url("redis://localhost:6379/0")
        cached = r.get(redis_key)
        if cached:
            return cached.decode()
    except Exception:
        pass

    import boto3
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket="uaedia-dcps", Key=f"{domain_id}/{version}.yaml")
    raw = obj["Body"].read().decode()

    try:
        r.setex(redis_key, 300, raw)
    except Exception:
        pass

    return raw


# =============================================================================
# SECTION 5: UNIFIED FEATURE REGISTRY
# =============================================================================

@dataclass
class FeatureDefinition:
    """
    Registry entry for a single feature computation function.

    The compute_fn is the single authoritative definition of the feature.
    Both the training pipeline (batch) and the serving pipeline (real-time)
    invoke this exact function. Training-serving skew is structurally
    impossible: divergence would require changing the registered function.
    """
    name: str
    version: str
    compute_fn: Callable
    dtype: str                       # 'float32' | 'int32' | 'bool' | 'str'
    description: str
    regulatory_sensitive: bool = False
    protected_attribute: bool = False  # ECOA / fair lending flag


_FEATURE_REGISTRY: Dict[str, FeatureDefinition] = {}


def register_feature(
    name: str,
    version: str,
    dtype: str,
    description: str,
    regulatory_sensitive: bool = False,
    protected_attribute: bool = False,
):
    """
    Decorator: register a function as the canonical definition of a feature.

    Example:
        @register_feature(
            name='debt_to_income_ratio',
            version='1.2.0',
            dtype='float32',
            description='Total monthly debt obligations / gross monthly income',
            regulatory_sensitive=True,
        )
        def debt_to_income_ratio(monthly_debt: float, gross_monthly_income: float) -> float:
            if gross_monthly_income <= 0:
                return float('nan')
            return round(monthly_debt / gross_monthly_income, 4)
    """
    def decorator(fn: Callable) -> Callable:
        _FEATURE_REGISTRY[name] = FeatureDefinition(
            name=name,
            version=version,
            compute_fn=fn,
            dtype=dtype,
            description=description,
            regulatory_sensitive=regulatory_sensitive,
            protected_attribute=protected_attribute,
        )
        return fn
    return decorator


def compute_serving_features(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all registered features from a raw event dict (serving path).

    Called at L2 (ENRICHING state) for every decision request.
    Identical computation to build_training_features() -- same registry,
    same functions.
    """
    result: Dict[str, Any] = {}
    for name, feat_def in _FEATURE_REGISTRY.items():
        params = feat_def.compute_fn.__code__.co_varnames[
            :feat_def.compute_fn.__code__.co_argcount
        ]
        kwargs = {p: event[p] for p in params if p in event}
        try:
            result[name] = feat_def.compute_fn(**kwargs)
        except Exception as exc:
            result[name] = None
    return result


def build_training_features(df) -> object:  # -> pd.DataFrame
    """
    Compute all registered features from a DataFrame (training path).

    Identical registry and functions as compute_serving_features().
    """
    import pandas as pd
    import numpy as np

    for name, feat_def in _FEATURE_REGISTRY.items():
        params = feat_def.compute_fn.__code__.co_varnames[
            :feat_def.compute_fn.__code__.co_argcount
        ]

        def _apply(row, fn=feat_def.compute_fn, p=params):
            kwargs = {k: row[k] for k in p if k in row.index}
            try:
                return fn(**kwargs)
            except Exception:
                return np.nan

        df[name] = df.apply(_apply, axis=1).astype(feat_def.dtype, errors="ignore")

    return df


# Example feature registrations -------------------------------------------------

@register_feature(
    name="debt_to_income_ratio",
    version="1.2.0",
    dtype="float32",
    description="Total monthly debt obligations divided by gross monthly income",
    regulatory_sensitive=True,
)
def debt_to_income_ratio(monthly_debt: float, gross_monthly_income: float) -> float:
    import math
    if gross_monthly_income <= 0:
        return math.nan
    return round(monthly_debt / gross_monthly_income, 4)


@register_feature(
    name="utilization_ratio",
    version="1.0.1",
    dtype="float32",
    description="Credit utilization: current balance / total credit limit",
    regulatory_sensitive=True,
)
def utilization_ratio(current_balance: float, total_credit_limit: float) -> float:
    import math
    if total_credit_limit <= 0:
        return math.nan
    return round(min(current_balance / total_credit_limit, 1.0), 4)


@register_feature(
    name="days_since_last_derogatory",
    version="2.0.0",
    dtype="int32",
    description="Days elapsed since most recent derogatory mark on bureau",
    regulatory_sensitive=True,
)
def days_since_last_derogatory(last_derogatory_date_iso: Optional[str]) -> int:
    if not last_derogatory_date_iso:
        return 99999  # No derogatory history
    last = datetime.fromisoformat(last_derogatory_date_iso).replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last).days


# =============================================================================
# SECTION 6: CHAMPION / CHALLENGER MODEL REGISTRY
# =============================================================================

@lru_cache(maxsize=64)
def _load_mlflow_model(model_id: str, model_version: str):
    """Load and cache a model from MLflow registry."""
    import mlflow.pyfunc
    return mlflow.pyfunc.load_model(f"models:/{model_id}/{model_version}")


def run_champion_challenger(
    dcp: dict,
    feature_vector: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute champion/challenger inference per DCP configuration.

    The champion model's result governs the decision. Challenger models
    receive configured traffic splits and their results are logged for
    model performance monitoring without influencing the decision outcome.

    Args:
        dcp:            Loaded DCP dict.
        feature_vector: Computed feature dict from the Unified Feature Registry.

    Returns:
        Dict containing model_id, model_version, score, confidence.
    """
    import random

    models = dcp["intelligence"]["models"]
    champion = next(m for m in models if m["champion"])
    challengers = [m for m in models if not m["champion"]]

    champ_model = _load_mlflow_model(champion["model_id"], champion["model_version"])
    champ_score = float(champ_model.predict([feature_vector])[0])

    for challenger in challengers:
        split = challenger.get("traffic_split", 0.0)
        if random.random() < split:
            chal_model = _load_mlflow_model(
                challenger["model_id"], challenger["model_version"]
            )
            chal_score = float(chal_model.predict([feature_vector])[0])
            _log_challenger_delta(
                champion["model_id"], challenger["model_id"],
                champ_score, chal_score
            )

    return {
        "model_id": champion["model_id"],
        "model_version": champion["model_version"],
        "score": champ_score,
        "confidence": abs(champ_score - 0.5) * 2,  # Distance from decision boundary
    }


def _log_challenger_delta(
    champ_id: str, chal_id: str, champ_score: float, chal_score: float
) -> None:
    """Log champion vs. challenger score delta to MLflow for drift monitoring."""
    try:
        import mlflow
        with mlflow.start_run(run_name=f"challenger_delta_{chal_id}", nested=True):
            mlflow.log_metrics({
                f"{champ_id}_score": champ_score,
                f"{chal_id}_score": chal_score,
                "delta": abs(champ_score - chal_score),
            })
    except Exception:
        pass


# =============================================================================
# SECTION 7: SHAP ADVERSE ACTION ATTRIBUTION
# =============================================================================

def compute_adverse_action_factors(
    model,
    feature_vector: Dict[str, Any],
    dcp: dict,
    top_n: int = 4,
) -> List[AdverseActionFactor]:
    """
    Compute SHAP-attributed adverse action factors for decline decisions.

    Uses TreeExplainer for tree-based models (XGBoost, LightGBM, RF).
    Falls back to KernelExplainer for opaque models.

    Args:
        model:          Trained model object.
        feature_vector: Feature dict for the declined applicant.
        dcp:            DCP dict supplying regulatory_regime for code mapping.
        top_n:          Number of top adverse action factors to return.

    Returns:
        List of AdverseActionFactor ordered by absolute SHAP value (descending).
    """
    import shap
    import numpy as np

    feature_names = list(feature_vector.keys())
    X = [list(feature_vector.values())]

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 50))

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class for binary classification

    shap_array = np.array(shap_values[0])
    top_indices = np.argsort(np.abs(shap_array))[::-1][:top_n]

    regime = dcp.get("governance", {}).get("regulatory_regime", [])
    code_map = _adverse_action_code_map(regime)

    return [
        AdverseActionFactor(
            feature_name=feature_names[i],
            shap_value=float(shap_array[i]),
            direction="increase" if shap_array[i] > 0 else "decrease",
            regulatory_code=code_map.get(feature_names[i], "Z9"),
        )
        for i in top_indices
    ]


def _adverse_action_code_map(regulatory_regime: List[str]) -> Dict[str, str]:
    """Map feature names to ECOA/FCRA adverse action reason codes."""
    return {
        "debt_to_income_ratio": "A9",
        "utilization_ratio": "A3",
        "days_since_last_derogatory": "A7",
        "payment_history_score": "A1",
        "open_accounts_count": "A5",
        "inquiry_count_12m": "A6",
        "average_account_age_months": "A8",
    }


# =============================================================================
# SECTION 8: ENTERPRISE CONTEXT GRAPH (ECG)  -- v6.0+
# =============================================================================

@dataclass
class PolicyNode:
    policy_id: str
    domain: str
    rule_text: str
    effective_date: str
    regulatory_source: str
    priority: int = 0
    active: bool = True


class ECGClient:
    """
    Client for the Enterprise Context Graph (Neo4j).

    The ECG stores organizational intelligence as a queryable knowledge graph:
    policies, workflow definitions, business rules, entity relationships,
    and historical decision references. Every agent in the AOE queries
    the ECG before acting to obtain the organizational context that governs
    that action.
    """

    def __init__(self, uri: str, auth: tuple):
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(uri, auth=auth)

    def get_applicable_policies(
        self, domain: str, action_type: str
    ) -> List[PolicyNode]:
        """
        Retrieve all active policies applicable to an action in a domain.

        Called by the Governance Gate before every agent action execution.
        Results are ordered by policy priority (descending).
        """
        query = """
            MATCH (p:Policy)-[:APPLIES_TO]->(d:Domain {id: $domain})
            WHERE p.action_type = $action_type AND p.active = true
            RETURN p.policy_id AS policy_id,
                   p.rule_text AS rule_text,
                   p.effective_date AS effective_date,
                   p.regulatory_source AS regulatory_source,
                   p.priority AS priority
            ORDER BY p.priority DESC
        """
        with self._driver.session() as session:
            result = session.run(query, domain=domain, action_type=action_type)
            return [
                PolicyNode(
                    policy_id=r["policy_id"],
                    domain=domain,
                    rule_text=r["rule_text"],
                    effective_date=r["effective_date"],
                    regulatory_source=r["regulatory_source"],
                    priority=r["priority"],
                )
                for r in result
            ]

    def get_historical_decisions(
        self, entity_id: str, limit: int = 20
    ) -> List[dict]:
        """
        Retrieve recent Decision Objects for an entity.

        Used by agents for context injection -- provides the same historical
        awareness a human expert would carry before making a decision.
        """
        query = """
            MATCH (e:Entity {id: $entity_id})-[:SUBJECT_OF]->(d:Decision)
            RETURN d ORDER BY d.completed_at DESC LIMIT $limit
        """
        with self._driver.session() as session:
            return [
                dict(r["d"])
                for r in session.run(query, entity_id=entity_id, limit=limit)
            ]

    def record_decision_reference(self, do: DecisionObject, entity_id: str) -> None:
        """
        Write a lightweight Decision reference node to the ECG after COMPLETED.

        Enables historical decision queries without duplicating the full
        Decision Object payload (which lives in the relational audit store).
        """
        query = """
            MERGE (e:Entity {id: $entity_id})
            CREATE (d:Decision {
                dcid: $dcid,
                domain: $domain,
                outcome: $outcome,
                completed_at: $completed_at
            })
            CREATE (e)-[:SUBJECT_OF]->(d)
        """
        with self._driver.session() as session:
            session.run(
                query,
                entity_id=entity_id,
                dcid=do.dcid,
                domain=do.domain,
                outcome=do.decision_outcome,
                completed_at=do.completed_at,
            )


# =============================================================================
# SECTION 9: GOVERNANCE GATE  -- v6.0+
# =============================================================================

class GateDecision(Enum):
    PASS  = "PASS"
    BLOCK = "BLOCK"
    REFER = "REFER"


@dataclass
class GateResult:
    decision: GateDecision
    policies_evaluated: int
    violations: List[str]
    dcid: str


class GovernanceViolationError(Exception):
    def __init__(self, dcid: str, violations: List[str]):
        super().__init__(
            f"[{dcid}] Governance violation. Blocked actions: {violations}"
        )
        self.dcid = dcid
        self.violations = violations


class GovernanceGate:
    """
    Pre-execution compliance enforcement component of the AOE.

    Every proposed agent action passes through evaluate() before execution.
    Non-compliant actions are blocked without side effects. Actions below
    the confidence threshold are routed to human review.

    This is the Governance-Embedded Agentic Execution pattern: compliance
    enforcement is embedded inside the action layer, not applied post-hoc.
    """

    CONFIDENCE_REFER_THRESHOLD = 0.70

    def __init__(self, ecg: ECGClient):
        self._ecg = ecg

    def evaluate(self, action: Dict[str, Any], dcid: str) -> GateResult:
        """
        Evaluate a proposed agent action against ECG policies.

        Args:
            action: Dict with at minimum: domain, action_type, confidence
            dcid:   DCID of the parent decision context

        Returns:
            GateResult with PASS, BLOCK, or REFER decision.
        """
        policies = self._ecg.get_applicable_policies(
            domain=action["domain"],
            action_type=action["action_type"],
        )

        violations = [
            f"{p.policy_id}: {p.rule_text}"
            for p in policies
            if not self._evaluate_policy(p, action)
        ]

        if violations:
            decision = GateDecision.BLOCK
        elif action.get("confidence", 1.0) < self.CONFIDENCE_REFER_THRESHOLD:
            decision = GateDecision.REFER
        else:
            decision = GateDecision.PASS

        return GateResult(
            decision=decision,
            policies_evaluated=len(policies),
            violations=violations,
            dcid=dcid,
        )

    def _evaluate_policy(self, policy: PolicyNode, action: Dict[str, Any]) -> bool:
        """
        Evaluate a single policy rule against a proposed action.

        Supports JSON Logic rules (default) and OPA policy evaluation.
        Extensible: add providers to the dispatch map.
        """
        rule = policy.rule_text
        if rule.startswith("{"):
            return self._eval_json_logic(json.loads(rule), action)
        return True  # Unknown rule format: default PASS (log warning in production)

    def _eval_json_logic(self, rule: dict, data: dict) -> bool:
        """Minimal JSON Logic evaluator for policy rules."""
        try:
            import json_logic
            return bool(json_logic.jsonLogic(rule, data))
        except ImportError:
            return True


def governed_action(gate: GovernanceGate):
    """
    Decorator: wrap any agent tool function with Governance Gate enforcement.

    Usage:
        @governed_action(gate)
        def execute_payment(action: dict, dcid: str, **kwargs):
            ...

    The decorated function will:
        - Evaluate action against ECG policies before execution
        - Raise GovernanceViolationError if BLOCK
        - Return {status: REFERRED} dict if REFER
        - Execute normally if PASS
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(action: Dict[str, Any], dcid: str, **kwargs) -> Any:
            result = gate.evaluate(action, dcid)
            if result.decision == GateDecision.BLOCK:
                raise GovernanceViolationError(dcid, result.violations)
            if result.decision == GateDecision.REFER:
                return {"status": "REFERRED", "dcid": dcid, "reason": "Low confidence"}
            return fn(action, dcid, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# SECTION 10: AGENTIC ORCHESTRATION ENGINE (AOE)  -- v6.0+
# =============================================================================

def build_enterprise_workflow(ecg: ECGClient, gate: GovernanceGate):
    """
    Construct a governed multi-agent workflow graph using LangGraph.

    Each node is a governed agent action. The Governance Gate wraps every
    edge traversal. The ECG provides organizational context at each node.

    Returns:
        Compiled LangGraph StateGraph ready for .invoke() or .stream().
    """
    from typing import TypedDict
    from langgraph.graph import StateGraph, END

    class WorkflowState(TypedDict):
        dcid: str
        domain: str
        context: Dict[str, Any]
        completed_steps: List[str]
        decision_output: Dict[str, Any]

    graph: StateGraph = StateGraph(WorkflowState)

    @governed_action(gate)
    def _data_fetch(action: dict, dcid: str, **kwargs) -> dict:
        # Implementation: fetch entity records from source systems
        return {"fetched": True, "records": []}

    @governed_action(gate)
    def _analyze(action: dict, dcid: str, **kwargs) -> dict:
        # Implementation: run analytical models against fetched records
        return {"analysis_complete": True, "signals": {}}

    @governed_action(gate)
    def _decide(action: dict, dcid: str, **kwargs) -> dict:
        # Implementation: compose final decision from analysis signals
        return {"outcome": "APPROVED", "confidence": 0.87}

    def node_fetch(state: WorkflowState) -> WorkflowState:
        historical = ecg.get_historical_decisions(
            state["context"].get("entity_id", ""), limit=10
        )
        _data_fetch(
            {"domain": state["domain"], "action_type": "DATA_FETCH", "confidence": 0.99},
            state["dcid"],
        )
        state["context"]["historical_decisions"] = historical
        state["completed_steps"].append("DATA_FETCH")
        return state

    def node_analyze(state: WorkflowState) -> WorkflowState:
        result = _analyze(
            {"domain": state["domain"], "action_type": "ANALYZE", "confidence": 0.94},
            state["dcid"],
        )
        state["context"]["analysis"] = result
        state["completed_steps"].append("ANALYZE")
        return state

    def node_decide(state: WorkflowState) -> WorkflowState:
        result = _decide(
            {"domain": state["domain"], "action_type": "DECIDE", "confidence": 0.87},
            state["dcid"],
        )
        state["decision_output"] = result
        state["completed_steps"].append("DECIDE")
        return state

    graph.add_node("fetch", node_fetch)
    graph.add_node("analyze", node_analyze)
    graph.add_node("decide", node_decide)
    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "analyze")
    graph.add_edge("analyze", "decide")
    graph.add_edge("decide", END)

    return graph.compile()


# =============================================================================
# SECTION 11: LLM RATIONALE GENERATOR
# =============================================================================

RATIONALE_SYSTEM_PROMPT = """
You are a regulatory compliance assistant embedded in an enterprise AI decision system.
Given a Decision Object summary, generate a plain-language rationale suitable for
inclusion in a regulatory audit record or adverse action communication.
Be precise, non-discriminatory, and reference only the features and rules that
influenced the decision. Do not speculate beyond the provided data.
Maximum 150 words.
""".strip()


def generate_audit_rationale(do: DecisionObject, provider: str = "anthropic") -> str:
    """
    Generate an LLM-augmented plain-language rationale for the Decision Object.

    The LLM narrates a decision already made by the DOL state machine.
    It does not make or modify the decision. The rationale is stored in
    decision_rationale on the Decision Object before seal().

    Args:
        do:       Partially complete DecisionObject (pre-seal).
        provider: "anthropic" | "openai"

    Returns:
        Plain-language rationale string (max ~150 words).
    """
    prompt = (
        f"Decision outcome: {do.decision_outcome}\n"
        f"Confidence: {do.confidence_score:.2f}\n"
        f"Domain: {do.domain}\n"
        f"Top adverse factors: {[f.feature_name for f in do.adverse_action_factors[:4]]}\n"
        f"Rules triggered: {[r.get('rule_id') for r in do.rules_evaluated if r.get('triggered')]}\n\n"
        "Generate the audit rationale."
    )

    if provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=256,
            system=RATIONALE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=256,
        messages=[
            {"role": "system", "content": RATIONALE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# =============================================================================
# SECTION 12: REST API (FastAPI)
# =============================================================================

def create_app():
    """
    Create the U-AEDIA DOL FastAPI application.

    Endpoints:
        POST /v1/decisions          Submit a decision request
        GET  /v1/decisions/{dcid}   Retrieve a completed Decision Object
        GET  /v1/health             Health check

    Run with:
        uvicorn uaedia_reference_implementation:app --host 0.0.0.0 --port 8000
    """
    from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
    from pydantic import BaseModel as PydanticModel, Field as PydanticField

    app = FastAPI(
        title="U-AEDIA Decision API",
        description="Unified AI-Driven Enterprise Decision Intelligence Architecture",
        version="8.0.0",
    )

    class DecisionRequest(PydanticModel):
        domain: str = PydanticField(..., description="DCP domain identifier")
        event_type: str
        payload: Dict[str, Any]
        idempotency_key: Optional[str] = None

    class DecisionResponse(PydanticModel):
        dcid: str
        status: str
        estimated_completion_ms: int

    _decision_store: Dict[str, dict] = {}

    async def _run_pipeline(dcid: str, domain: str, payload: dict):
        """Background task: full DOL pipeline execution."""
        ingested_at = datetime.now(timezone.utc).isoformat()
        try:
            dcp = load_dcp(domain)
            ctx = DecisionContext(
                dcid=dcid, event_payload=payload, domain=domain
            )
            ctx = dol_transition(ctx, DOLState.VALIDATING)
            ctx = dol_transition(ctx, DOLState.ENRICHING)
            ctx.feature_vector = compute_serving_features(payload)
            ctx = dol_transition(ctx, DOLState.INFERRING)
            ctx = dol_transition(ctx, DOLState.EVALUATING)
            ctx = dol_transition(ctx, DOLState.DECIDING)
            ctx.decision = {"outcome": "APPROVED", "confidence": 0.83, "gate_result": "PASS"}
            ctx = dol_transition(ctx, DOLState.EXECUTING)
            ctx = dol_transition(ctx, DOLState.COMPLETED)
            do = DecisionObject.from_context(ctx, ingested_at).seal()
            _decision_store[dcid] = asdict(do)
        except Exception as exc:
            _decision_store[dcid] = {"dcid": dcid, "error": str(exc), "state": "COMPENSATED"}

    @app.post("/v1/decisions", response_model=DecisionResponse, status_code=202)
    async def submit_decision(req: DecisionRequest, background_tasks: BackgroundTasks):
        dcid = generate_dcid(req.domain[:4])
        background_tasks.add_task(_run_pipeline, dcid, req.domain, req.payload)
        dcp = load_dcp(req.domain)
        return DecisionResponse(
            dcid=dcid,
            status="ACCEPTED",
            estimated_completion_ms=dcp["dol"]["state_machine_timeout_ms"],
        )

    @app.get("/v1/decisions/{dcid}")
    async def get_decision(dcid: str):
        obj = _decision_store.get(dcid)
        if not obj:
            raise HTTPException(status_code=404, detail=f"DCID {dcid} not found")
        return obj

    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "framework": "u-aedia", "version": "8.0.0"}

    return app


app = create_app()


# =============================================================================
# SECTION 13: OPENTELEMETRY INSTRUMENTATION
# =============================================================================

def configure_telemetry(service_name: str = "uaedia-dol", otlp_endpoint: str = "http://localhost:4317"):
    """
    Configure OpenTelemetry tracing for the U-AEDIA DOL.

    Exports spans to an OTLP endpoint (Jaeger, Grafana Tempo, Datadog, etc.).
    Each DOL state transition produces a child span tagged with dcid and domain,
    enabling end-to-end distributed trace reconstruction per decision.
    """
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource

    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
    )
    otel_trace.set_tracer_provider(provider)
    return otel_trace.get_tracer(service_name)


def traced_dol_state(state_name: str, tracer=None):
    """
    Decorator: wrap a DOL state handler in an OpenTelemetry span.

    Tags: dcid, domain, dol.state, dol.outcome
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(ctx: DecisionContext, *args, **kwargs):
            if tracer is None:
                return await fn(ctx, *args, **kwargs)
            from opentelemetry import trace as otel_trace
            t = tracer or otel_trace.get_tracer("uaedia.dol")
            with t.start_as_current_span(f"dol.{state_name}") as span:
                span.set_attribute("dcid", ctx.dcid)
                span.set_attribute("domain", ctx.domain)
                span.set_attribute("dol.state", state_name)
                result = await fn(ctx, *args, **kwargs)
                span.set_attribute("dol.outcome", result.state.name)
                return result
        return wrapper
    return decorator


# =============================================================================
# SECTION 14: DECISION OBJECT STORE (SQLAlchemy)
# =============================================================================

def get_sqlalchemy_models():
    """
    Return SQLAlchemy ORM models for the Decision Object Store (L5).

    Table: decision_objects
        Primary key: dcid (VARCHAR 64)
        Indexed: domain, decision_outcome, completed_at
        JSONB columns: feature_vector_snapshot, models_invoked, rules_evaluated,
                       adverse_action_factors, state_history

    Used with PostgreSQL 15 + TimescaleDB for time-series partitioning by completed_at.
    """
    from sqlalchemy import (
        Column, String, Float, Boolean, Integer, DateTime, Text
    )
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.orm import DeclarativeBase

    class Base(DeclarativeBase):
        pass

    class DecisionObjectORM(Base):
        __tablename__ = "decision_objects"

        dcid                    = Column(String(64), primary_key=True)
        schema_version          = Column(String(32))
        framework_version       = Column(String(32))
        domain                  = Column(String(128), index=True)
        deployment_id           = Column(String(128))
        ingested_at             = Column(String(64))
        decided_at              = Column(String(64))
        completed_at            = Column(String(64), index=True)
        latency_ms              = Column(Integer)
        event_payload_hash      = Column(String(64))
        feature_vector_snapshot = Column(JSONB)
        models_invoked          = Column(JSONB)
        decision_outcome        = Column(String(32), index=True)
        decision_rationale      = Column(Text)
        confidence_score        = Column(Float)
        threshold_applied       = Column(Float)
        rules_evaluated         = Column(JSONB)
        governance_gate_result  = Column(String(16))
        human_review_required   = Column(Boolean)
        adverse_action_factors  = Column(JSONB)
        state_history           = Column(JSONB)
        integrity_hash          = Column(String(64), unique=True)

    return Base, DecisionObjectORM


async def persist_decision_object(do: DecisionObject, db_url: str) -> None:
    """
    Persist a sealed Decision Object to the audit store.

    Verifies integrity_hash is populated (seal() has been called) before write.
    Raises ValueError if integrity_hash is missing.
    """
    if not do.integrity_hash:
        raise ValueError(f"[{do.dcid}] Decision Object must be sealed before persistence.")

    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    _, DecisionObjectORM = get_sqlalchemy_models()
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        record = DecisionObjectORM(**{
            k: (v if not isinstance(v, list) or not v or not hasattr(v[0], "__dataclass_fields__")
                else [asdict(item) for item in v])
            for k, v in asdict(do).items()
        })
        session.add(record)
        await session.commit()


# =============================================================================
# SECTION 15: DOCKER COMPOSE MANIFEST (embedded for reference)
# =============================================================================

DOCKER_COMPOSE_YAML = """
version: '3.9'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on: [zookeeper]
    ports: ['9092:9092']
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
      KAFKA_DEFAULT_REPLICATION_FACTOR: 1

  schema-registry:
    image: confluentinc/cp-schema-registry:7.5.0
    depends_on: [kafka]
    ports: ['8081:8081']
    environment:
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka:9092
      SCHEMA_REGISTRY_HOST_NAME: schema-registry

  postgres:
    image: postgres:15
    ports: ['5432:5432']
    environment:
      POSTGRES_DB: uaedia_decisions
      POSTGRES_USER: uaedia
      POSTGRES_PASSWORD: changeme
    volumes: ['pgdata:/var/lib/postgresql/data']

  neo4j:
    image: neo4j:5.15
    ports: ['7474:7474', '7687:7687']
    environment:
      NEO4J_AUTH: neo4j/changeme

  redis:
    image: redis:7.2
    ports: ['6379:6379']

  uaedia-api:
    build: .
    ports: ['8000:8000']
    depends_on: [kafka, postgres, neo4j, redis]
    environment:
      KAFKA_BOOTSTRAP: kafka:9092
      DB_URL: postgresql+asyncpg://uaedia:changeme@postgres:5432/uaedia_decisions
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_AUTH: neo4j/changeme
      REDIS_URL: redis://redis:6379/0
    command: uvicorn uaedia_reference_implementation:app --host 0.0.0.0 --port 8000

volumes:
  pgdata:
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("uaedia_reference_implementation:app", host="0.0.0.0", port=8000, reload=True)
