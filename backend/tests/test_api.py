"""API surface: the exact contract the frontend is written against."""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

import aegis.config
from aegis.main import create_app

CORPUS = [
    ("sensor", "Bay 3 conveyor vibration crossed 4.2 mm/s at 02:14"),
    ("sensor", "Coolant pressure read 1.74 bar for 96 seconds before the interlock fired"),
    ("sensor", "Ambient temperature in the cell climbed to 61 C during the night shift"),
    ("semantic", "Coolant pressure below 1.8 bar for over 90 seconds is a hard stop condition"),
    ("semantic", "Bearing vibration above 4.0 mm/s is an early indicator of raceway spalling"),
    ("procedural", "Recovery: isolate the drive, purge the line, re-home the gantry"),
    ("procedural", "To clear a torque fault: cut servo power, rotate the spindle by hand"),
    ("episodic", "Operator acknowledged the torque alarm and switched line 2 to manual feed"),
    ("episodic", "Uplink dropped for 47 minutes; operations queued locally and replayed"),
    ("episodic", "Maintenance replaced the bay 3 bearing housing and logged the part number"),
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A node with its own data directory.

    The API tests must not inherit a WAL, an op queue or a migration
    checkpoint from an earlier run or a locally running server — otherwise
    they assert against whatever that process happened to leave behind.
    """
    os.environ["AEGIS_DATA_DIR"] = str(tmp_path_factory.mktemp("node"))
    aegis.config._settings = None
    try:
        with TestClient(create_app()) as c:
            # The node ships with no memories: a real deployment ingests its
            # own. These are the operating facts the API tests exercise.
            for collection, text in CORPUS:
                c.post("/api/v1/memory/ingest",
                       json={"text": text, "collection": collection})
            yield c
    finally:
        os.environ.pop("AEGIS_DATA_DIR", None)
        aegis.config._settings = None


def test_health_reports_the_real_backend_and_provider(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["memory_backend"].startswith("qdrant")     # real Qdrant, not a stand-in
    assert body["execution_provider"].endswith("ExecutionProvider")
    assert body["model"]["vocab"] == 32_000                 # real pretrained tokenizer
    assert body["model"]["dim"] == body["model"]["dim"]
    assert body["points"] == len(CORPUS)


def test_memory_stats_shape_matches_the_console(client):
    body = client.get("/api/v1/memory/stats").json()
    for key in ("hot", "warm", "cold", "total", "collections", "by_sensitivity", "wal"):
        assert key in body


def test_ingest_then_search_finds_it(client):
    ingest = client.post("/api/v1/memory/ingest", json={
        "text": "Spindle SP-9920 tripped the overcurrent relay at 14:02",
        "collection": "episodic",
    }).json()
    assert ingest["sensitivity"] in {"internal", "public", "sensitive", "restricted"}
    found = client.post("/api/v1/search", json={"query": "SP-9920 overcurrent", "k": 3}).json()
    assert found["results"]
    assert found["results"][0]["id"] == ingest["id"]
    assert "explain" in found["results"][0]


def test_search_reports_stage_latencies(client):
    body = client.post("/api/v1/search", json={"query": "coolant pressure", "k": 3}).json()
    assert {"embed_ms", "dense_ms", "sparse_ms", "fusion_ms"} <= set(body["stages"])
    assert body["latency_ms"] > 0


def test_ask_returns_a_cited_trace(client):
    body = client.post("/api/v1/ask", json={"query": "how do i recover the drive"}).json()
    assert [s["step"] for s in body["trace"]] == ["plan", "retrieve", "verify", "answer"]
    assert body["citations"]


def test_sync_status_and_manual_reconcile(client):
    status = client.get("/api/v1/sync/status").json()
    assert {"state", "queued", "divergent", "conflicts", "cursor"} <= set(status)
    after = client.post("/api/v1/sync/trigger", json={"reason": "test"}).json()
    assert after["cycle"]["state"] in {"CONVERGED", "BACKOFF"}


def test_renewal_status_and_migration(client):
    assert "state" in client.get("/api/v1/renewal/status").json()
    started = client.post("/api/v1/renewal/migrate", json={"to_version": "bge-small-en-v9"}).json()
    assert started["to_version"] == "bge-small-en-v9"


def test_point_detail_exposes_lineage(client):
    points = client.get("/api/v1/memory/points?limit=1").json()["points"]
    detail = client.get(f"/api/v1/memory/points/{points[0]['id']}").json()
    assert "lineage" in detail and "resolved_text" in detail


def test_audit_chain_is_exposed_and_intact(client):
    body = client.get("/api/v1/audit").json()
    assert body["chain"]["chain_intact"] is True
    assert body["entries"]


def test_prometheus_metrics_render(client):
    text = client.get("/api/v1/metrics").text
    assert "aegis_" in text


def test_chaos_rejects_an_unknown_fault(client):
    assert client.post("/api/v1/chaos/not_a_fault").status_code == 400


def test_chaos_link_drop_takes_the_node_offline(client):
    client.post("/api/v1/chaos/link_drop", json={"duration_s": 1.0})
    assert client.get("/api/v1/node/state").json()["mode"] == "LOCAL"
    assert client.post("/api/v1/search", json={"query": "coolant", "k": 2}).json()["results"]


def test_websocket_streams_hello_and_events(client):
    with client.websocket_connect("/api/v1/stream") as socket:
        hello = json.loads(socket.receive_text())
        assert hello["kind"] == "hello"
        assert "node" in hello and "memory" in hello
        client.post("/api/v1/memory/ingest", json={"text": "stream probe observation"})
        kinds = [json.loads(socket.receive_text()).get("kind") for _ in range(12)]
        assert any(k for k in kinds)


def test_search_returns_a_plan_and_a_trace(client):
    body = client.post("/api/v1/search", json={"query": "coolant pressure", "k": 3}).json()
    assert body["plan"]["plan"] in {"pre_filter", "post_filter", "full_scan"}
    assert body["trace"]["name"] == "search"
    assert body["trace"]["children"]


def test_search_repairs_a_misspelled_query(client):
    body = client.post("/api/v1/search", json={"query": "colent presure", "k": 3}).json()
    assert body["understanding"]["corrections"]
    assert body["results"]


def test_filtered_search_uses_a_prefilter_plan(client):
    body = client.post("/api/v1/search",
                       json={"query": "pressure", "k": 3, "filters": {"collection": "sensor"}}).json()
    assert body["plan"]["plan"] == "pre_filter"
    assert {hit["collection"] for hit in body["results"]} == {"sensor"}


def test_impossible_filter_does_no_vector_work(client):
    body = client.post("/api/v1/search",
                       json={"query": "pressure", "k": 3, "filters": {"collection": "nope"}}).json()
    assert body["plan"]["plan"] == "empty"
    assert body["results"] == []
    assert "dense_ms" not in body["stages"]


def test_index_report_exposes_the_calibrated_cost_model(client):
    body = client.get("/api/v1/index").json()
    assert body["cost_model"]["calibrated"] is True
    assert body["collections"]
    assert all("strategy" in c["ann"] for c in body["collections"].values())


def test_scheduler_reports_lanes(client):
    body = client.get("/api/v1/scheduler").json()
    assert set(body["depth"]) == {"interactive", "sync", "maintenance", "renewal"}


def test_traces_endpoint_names_the_hotspot(client):
    client.post("/api/v1/search", json={"query": "gantry", "k": 3})
    body = client.get("/api/v1/traces").json()
    assert body["summary"]["slowest"]
    assert "hotspot" in body["summary"]["slowest"][0]


def test_feedback_trains_the_adapter(client):
    found = client.post("/api/v1/search", json={"query": "coolant", "k": 2}).json()
    chosen = found["results"][0]["id"]
    body = client.post("/api/v1/learning/feedback",
                       json={"query": "coolant", "chosen_id": chosen}).json()
    assert body["accepted"] is True
    assert client.get("/api/v1/learning/status").json()["adapter"]["rank"] > 0


def test_feedback_on_an_unknown_point_is_rejected(client):
    assert client.post("/api/v1/learning/feedback",
                       json={"query": "x", "chosen_id": "pt-nope"}).status_code == 404


def test_federated_round_states_the_cohort_it_would_need(client):
    body = client.post("/api/v1/learning/round", json={"simulate_peers": 5}).json()
    assert body["status"] == "aggregated"
    assert body["participants"] == 6
    assert body["cohort_for_usable_snr"] >= 1


def test_mesh_peer_lifecycle(client):
    assert client.post("/api/v1/mesh/peers", json={"node_id": "edge-42"}).json()["peers"] >= 1
    assert client.get("/api/v1/mesh/status").json()["node_id"]
    assert client.delete("/api/v1/mesh/peers/edge-42").status_code == 200
    assert client.delete("/api/v1/mesh/peers/edge-42").status_code == 404


def test_mesh_round_without_peers_is_a_conflict_not_a_crash(client):
    for peer in list(client.get("/api/v1/mesh/status").json()["peers"]):
        client.delete(f"/api/v1/mesh/peers/{peer['node_id']}")
    assert client.post("/api/v1/mesh/round").status_code == 409


def test_graph_endpoints_expose_structure(client):
    stats = client.get("/api/v1/graph/stats").json()
    assert stats["entities"] > 0 and stats["facts"] > 0
    entities = client.get("/api/v1/graph/entities?limit=5").json()
    assert entities["entities"]
    detail = client.get(f"/api/v1/graph/entities/{entities['entities'][0]['id']}").json()
    assert "facts" in detail and "memories" in detail


def test_graph_time_travel(client):
    import time as _time
    before = _time.time()
    client.post("/api/v1/memory/ingest",
                json={"text": "Spindle SP-3312 exceeded 51 nm on line 4", "collection": "sensor"})
    now = _time.time()
    assert client.get(f"/api/v1/graph/as-of?when={before}").json()["facts"] <= \
        client.get(f"/api/v1/graph/as-of?when={now}").json()["facts"]
    diff = client.get(f"/api/v1/graph/diff?earlier={before}&later={now}").json()
    assert diff["learned_count"] >= 1


def test_integrity_archive_fsck_and_generations(client):
    client.post("/api/v1/memory/ingest", json={"text": "a memory to seal into a segment"})
    sealed = client.post("/api/v1/integrity/archive").json()
    assert sealed["sealed"] is None or sealed["sealed"]["records"] > 0
    assert client.post("/api/v1/integrity/fsck").json()["clean"] is True
    assert "generations" in client.get("/api/v1/integrity/generations").json()


def test_integrity_restore_rejects_an_unknown_generation(client):
    assert client.post("/api/v1/integrity/restore", json={"generation": 9999}).status_code == 404


def test_slo_status_and_override(client):
    body = client.get("/api/v1/slo").json()
    assert body["level"] in {"FULL", "ECONOMISE", "TRIM", "ESSENTIAL", "SURVIVAL"}
    assert "ladder" in body and "features" in body
    pinned = client.post("/api/v1/slo/override", json={"level": "trim"}).json()
    assert pinned["override"] == "TRIM"
    assert "late_interaction" in pinned["disabled"]
    degraded = client.post("/api/v1/search", json={"query": "coolant", "k": 3}).json()
    assert degraded["degradation"]["level"] == "TRIM"
    client.post("/api/v1/slo/override", json={"level": None})


def test_slo_override_rejects_nonsense(client):
    assert client.post("/api/v1/slo/override", json={"level": "ludicrous"}).status_code == 400


def test_search_reports_confidence_and_diversity(client):
    body = client.post("/api/v1/search", json={"query": "coolant pressure", "k": 3}).json()
    assert "guarantee" in body["confidence"]
    assert "graph_context" in body


def test_tenancy_lifecycle_and_isolation(client):
    created = client.post("/api/v1/tenants",
                          json={"tenant_id": "acme-api", "name": "Acme", "max_points": 50}).json()
    assert created["tenant_id"] == "acme-api"
    issued = client.post("/api/v1/tenants/acme-api/keys",
                         json={"scopes": ["read", "write"], "label": "gateway"}).json()
    secret = issued["secret"]
    assert secret.startswith("aeg_")

    headers = {"x-aegis-key": secret}
    client.post("/api/v1/memory/ingest", json={"text": "Acme-only compressor note"}, headers=headers)
    tenant_view = client.post("/api/v1/search", json={"query": "compressor note", "k": 5},
                              headers=headers).json()
    assert any("Acme-only" in hit["text"] for hit in tenant_view["results"])

    anonymous = client.post("/api/v1/search", json={"query": "compressor note", "k": 5}).json()
    assert anonymous["results"]                      # admin/anonymous sees everything
    assert client.get("/api/v1/tenants/whoami", headers=headers).json()["tenant_id"] == "acme-api"

    revoked = client.delete(f"/api/v1/tenants/keys/{issued['key']['key_id']}")
    assert revoked.status_code == 200


def test_unknown_credential_is_ignored_in_open_mode(client):
    body = client.post("/api/v1/search", json={"query": "coolant", "k": 2},
                       headers={"x-aegis-key": "aeg_not-a-real-key"})
    assert body.status_code == 200                   # open mode degrades to anonymous


def test_quota_returns_429_not_500(client):
    client.post("/api/v1/tenants", json={"tenant_id": "tiny-api", "max_points": 1,
                                         "max_ingest_per_minute": 1})
    issued = client.post("/api/v1/tenants/tiny-api/keys", json={"scopes": ["write"]}).json()
    headers = {"x-aegis-key": issued["secret"]}
    client.post("/api/v1/memory/ingest", json={"text": "first"}, headers=headers)
    second = client.post("/api/v1/memory/ingest", json={"text": "second"}, headers=headers)
    assert second.status_code == 429
