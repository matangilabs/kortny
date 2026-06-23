from sqlalchemy import CheckConstraint

from kortny.db.models import Base, LLMProvider, TaskEventType, TaskStatus


def test_mvp_schema_declares_all_core_tables() -> None:
    assert set(Base.metadata.tables) == {
        "installations",
        "encrypted_secrets",
        "tasks",
        "task_events",
        "workspace_state",
        "slack_identities",
        "slack_channel_memberships",
        "slack_inbound_events",
        "slack_side_effects",
        "schedules",
        "dashboard_users",
        "dashboard_oauth_states",
        "composio_connections",
        "autonomy_policies",
        "observe_policies",
        "observation_events",
        "observe_channel_profiles",
        "kg_entities",
        "kg_edges",
        "kg_evidence",
        "procedural_skills",
        "procedural_skill_versions",
        "procedural_skill_invocations",
        "skill_files",
        "skill_enablements",
        "mcp_servers",
        "mcp_server_tools",
        "tool_pins",
        "tool_embeddings",
        "interactive_actions",
        "consolidation_runs",
        "episodes",
        "llm_usage",
        "llm_budget_policies",
        "llm_config_audit",
        "llm_model_catalog",
        "llm_model_pricing",
        "llm_provider_accounts",
        "llm_tier_assignments",
        "witness_opportunity_candidates",
        "witness_delivery_log",
        "proactive_action_events",
        "artifacts",
        "model_pricing",
        "assistant_thread_context",
        "composio_tool_cards",
        "file_extraction_cache",
    }


def test_task_status_enum_matches_locked_schema() -> None:
    assert [status.value for status in TaskStatus] == [
        "pending",
        "running",
        "waiting_approval",
        "succeeded",
        "failed",
        "crashed",
        "cancelled",
    ]


def test_llm_provider_enum_matches_locked_schema() -> None:
    assert [provider.value for provider in LLMProvider] == [
        "openai",
        "anthropic",
        "openrouter",
    ]


def test_task_event_type_enum_matches_locked_schema() -> None:
    assert [event_type.value for event_type in TaskEventType] == [
        "task_created",
        "status_changed",
        "llm_call",
        "tool_call",
        "tool_result",
        "artifact_created",
        "message_posted",
        "error",
        "log",
    ]


def test_task_table_has_queue_and_thread_indexes() -> None:
    task_table = Base.metadata.tables["tasks"]
    index_names = {index.name for index in task_table.indexes}

    assert {"idx_tasks_claim", "idx_tasks_history", "idx_tasks_thread"} <= index_names


def test_workspace_state_table_has_memory_policy_constraints_and_indexes() -> None:
    workspace_state = Base.metadata.tables["workspace_state"]
    constraint_names = {constraint.name for constraint in workspace_state.constraints}
    index_names = {index.name for index in workspace_state.indexes}

    assert {
        "ck_workspace_state_scope_type",
        "ck_workspace_state_status",
        "ck_workspace_state_source_kind",
        "ck_workspace_state_scope_id",
    } <= constraint_names
    assert {
        "idx_workspace_state_active_unique",
        "idx_workspace_state_active_lookup",
        "idx_workspace_state_source",
        "idx_workspace_state_expires_at",
    } <= index_names


def test_episode_table_has_bounded_retrieval_indexes() -> None:
    episodes = Base.metadata.tables["episodes"]
    constraint_names = {constraint.name for constraint in episodes.constraints}
    index_names = {index.name for index in episodes.indexes}

    assert {"ck_episodes_outcome", "idx_episodes_task_unique"} <= constraint_names
    assert {
        "idx_episodes_thread",
        "idx_episodes_channel",
        "idx_episodes_user",
    } <= index_names


def test_slack_identity_table_has_lookup_constraints_and_indexes() -> None:
    identities = Base.metadata.tables["slack_identities"]
    constraint_names = {constraint.name for constraint in identities.constraints}
    index_names = {index.name for index in identities.indexes}

    assert {"ck_slack_identity_kind", "idx_slack_identity_unique"} <= constraint_names
    assert {"idx_slack_identity_lookup", "idx_slack_identity_seen"} <= index_names


def test_slack_channel_membership_table_has_presence_constraints_and_indexes() -> None:
    memberships = Base.metadata.tables["slack_channel_memberships"]
    constraint_names = {constraint.name for constraint in memberships.constraints}
    index_names = {index.name for index in memberships.indexes}

    assert {
        "ck_slack_channel_memberships_status",
        "ck_slack_channel_memberships_discovered_via",
        "ck_slack_channel_memberships_onboarding_status",
        "idx_slack_channel_memberships_unique",
    } <= constraint_names
    assert {
        "idx_slack_channel_memberships_lookup",
        "idx_slack_channel_memberships_status",
        "idx_slack_channel_memberships_onboarding",
    } <= index_names


def test_slack_side_effects_table_has_outbox_constraints_and_indexes() -> None:
    side_effects = Base.metadata.tables["slack_side_effects"]
    constraint_names = {constraint.name for constraint in side_effects.constraints}
    index_names = {index.name for index in side_effects.indexes}

    assert {
        "ck_slack_side_effects_operation",
        "ck_slack_side_effects_status",
        "idx_slack_side_effects_idempotency",
    } <= constraint_names
    assert {
        "idx_slack_side_effects_status",
        "idx_slack_side_effects_task",
        "idx_slack_side_effects_target",
    } <= index_names


def test_schedules_table_has_schedule_policy_constraints_and_indexes() -> None:
    schedules = Base.metadata.tables["schedules"]
    constraint_names = {constraint.name for constraint in schedules.constraints}
    index_names = {index.name for index in schedules.indexes}

    assert {
        "ck_schedules_owner_type",
        "ck_schedules_owner",
        "ck_schedules_spec_kind",
        "ck_schedules_spec",
        "ck_schedules_catchup_policy",
        "ck_schedules_catchup_window",
        "ck_schedules_overlap_policy",
        "ck_schedules_status",
        "ck_schedules_cost_ceiling",
    } <= constraint_names
    assert {
        "idx_schedules_due",
        "idx_schedules_owner",
        "idx_schedules_status",
    } <= index_names


def test_observe_channel_profile_table_has_staleness_constraints_and_indexes() -> None:
    profiles = Base.metadata.tables["observe_channel_profiles"]
    constraint_names = {constraint.name for constraint in profiles.constraints}
    index_names = {index.name for index in profiles.indexes}

    assert {
        "ck_observe_channel_profiles_status",
        "ck_observe_channel_profiles_fresh_window",
        "ck_observe_channel_profiles_archive_window",
        "idx_observe_channel_profiles_unique",
    } <= constraint_names
    assert {
        "idx_observe_channel_profiles_lookup",
        "idx_observe_channel_profiles_last_profiled",
        "idx_observe_channel_profiles_source_task",
    } <= index_names


def test_composio_connection_table_has_visibility_constraints_and_indexes() -> None:
    connections = Base.metadata.tables["composio_connections"]
    constraint_names = {constraint.name for constraint in connections.constraints}
    index_names = {index.name for index in connections.indexes}

    assert {
        "ck_composio_connections_visibility_scope_type",
        "ck_composio_connections_status",
        "ck_composio_connections_visibility_scope_id",
    } <= constraint_names
    assert {
        "idx_composio_connections_connected_account",
        "idx_composio_connections_allowed_lookup",
        "idx_composio_connections_owner",
        "idx_composio_connections_toolkit",
    } <= index_names


def test_procedural_skill_tables_have_scope_version_and_invocation_indexes() -> None:
    skills = Base.metadata.tables["procedural_skills"]
    versions = Base.metadata.tables["procedural_skill_versions"]
    invocations = Base.metadata.tables["procedural_skill_invocations"]

    skill_constraints = {constraint.name for constraint in skills.constraints}
    skill_indexes = {index.name for index in skills.indexes}
    version_constraints = {constraint.name for constraint in versions.constraints}
    version_indexes = {index.name for index in versions.indexes}
    invocation_indexes = {index.name for index in invocations.indexes}

    assert {
        "ck_procedural_skills_owner_type",
        "ck_procedural_skills_status",
        "ck_procedural_skills_trust_level",
        "ck_procedural_skills_visibility",
        "ck_procedural_skills_owner_id",
    } <= skill_constraints
    assert {
        "idx_procedural_skills_unique_slug",
        "idx_procedural_skills_catalog",
    } <= skill_indexes
    assert {
        "ck_procedural_skill_versions_status",
        "idx_procedural_skill_versions_unique",
    } <= version_constraints
    assert {
        "idx_procedural_skill_versions_active",
        "idx_procedural_skill_versions_tags",
        "idx_procedural_skill_versions_modes",
    } <= version_indexes
    assert {
        "idx_procedural_skill_invocations_task",
        "idx_procedural_skill_invocations_skill",
        "idx_procedural_skill_invocations_installation",
    } <= invocation_indexes


def test_observe_tables_have_policy_and_event_constraints() -> None:
    policies = Base.metadata.tables["observe_policies"]
    events = Base.metadata.tables["observation_events"]

    policy_constraints = {constraint.name for constraint in policies.constraints}
    policy_indexes = {index.name for index in policies.indexes}
    event_constraints = {constraint.name for constraint in events.constraints}
    event_indexes = {index.name for index in events.indexes}

    assert {
        "ck_observe_policies_scope_type",
        "ck_observe_policies_observation_status",
        "ck_observe_policies_proactivity_status",
        "ck_observe_policies_scope_id",
    } <= policy_constraints
    assert {
        "idx_observe_policies_scope_unique",
        "idx_observe_policies_lookup",
    } <= policy_indexes
    assert {"ck_observation_events_event_type"} <= event_constraints
    assert {
        "idx_observation_events_event_unique",
        "idx_observation_events_channel",
        "idx_observation_events_user",
        "idx_observation_events_purged",
    } <= event_indexes


def test_knowledge_graph_tables_have_scope_lifecycle_and_evidence_indexes() -> None:
    entities = Base.metadata.tables["kg_entities"]
    edges = Base.metadata.tables["kg_edges"]
    evidence = Base.metadata.tables["kg_evidence"]

    entity_constraints = {constraint.name for constraint in entities.constraints}
    entity_indexes = {index.name for index in entities.indexes}
    edge_constraints = {constraint.name for constraint in edges.constraints}
    edge_indexes = {index.name for index in edges.indexes}
    evidence_constraints = {constraint.name for constraint in evidence.constraints}
    evidence_indexes = {index.name for index in evidence.indexes}

    assert {
        "ck_kg_entities_type",
        "ck_kg_entities_visibility_scope_type",
        "ck_kg_entities_visibility_scope_id",
        "ck_kg_entities_source_type",
        "ck_kg_entities_lifecycle_state",
        "ck_kg_entities_confidence_score",
    } <= entity_constraints
    assert {
        "idx_kg_entities_current_unique_key",
        "idx_kg_entities_lookup",
        "idx_kg_entities_scope",
        "idx_kg_entities_external_ref",
        "idx_kg_entities_attrs",
    } <= entity_indexes

    assert {
        "ck_kg_edges_relationship_type",
        "ck_kg_edges_visibility_scope_type",
        "ck_kg_edges_visibility_scope_id",
        "ck_kg_edges_source_type",
        "ck_kg_edges_lifecycle_state",
        "ck_kg_edges_confidence_score",
    } <= edge_constraints
    assert {
        "idx_kg_edges_current_unique",
        "idx_kg_edges_source_lookup",
        "idx_kg_edges_target_lookup",
        "idx_kg_edges_scope",
        "idx_kg_edges_attrs",
    } <= edge_indexes

    assert {
        "ck_kg_evidence_target_kind",
        "ck_kg_evidence_source_type",
        "ck_kg_evidence_consensus_count",
        "ck_kg_evidence_confidence_score",
        "ck_kg_evidence_source_reference",
    } <= evidence_constraints
    assert {
        "idx_kg_evidence_target",
        "idx_kg_evidence_task",
        "idx_kg_evidence_episode",
        "idx_kg_evidence_observation",
        "idx_kg_evidence_slack_message",
    } <= evidence_indexes


def test_memory_spine_schema_declares_bitemporal_columns_and_runs_table() -> None:
    entities = Base.metadata.tables["kg_entities"]
    edges = Base.metadata.tables["kg_edges"]
    for table in (entities, edges):
        columns = set(table.columns.keys())
        assert {"valid_at", "invalid_at", "system_expired_at"} <= columns
        assert table.columns["valid_at"].nullable is True
        assert table.columns["invalid_at"].nullable is True
        assert table.columns["system_expired_at"].nullable is True
    assert "idx_kg_entities_invalid_at" in {index.name for index in entities.indexes}
    assert "idx_kg_edges_invalid_at" in {index.name for index in edges.indexes}

    runs = Base.metadata.tables["consolidation_runs"]
    assert {
        "id",
        "installation_id",
        "started_at",
        "finished_at",
        "status",
        "counters_json",
        "cost_usd",
        "error",
    } <= set(runs.columns.keys())
    assert "ck_consolidation_runs_status" in {
        constraint.name for constraint in runs.constraints
    }
    assert "idx_consolidation_runs_installation_started" in {
        index.name for index in runs.indexes
    }

    embeddings = Base.metadata.tables["tool_embeddings"]
    kind_check = next(
        constraint
        for constraint in embeddings.constraints
        if isinstance(constraint, CheckConstraint)
        and constraint.name == "ck_tool_embeddings_kind"
    )
    for kind in ("tool_card", "skill", "fact", "episode", "kg_entity"):
        assert kind in str(kind_check.sqltext)


def test_llm_usage_table_has_cache_token_columns() -> None:
    columns = Base.metadata.tables["llm_usage"].columns
    assert "cache_creation_input_tokens" in columns
    assert "cache_read_input_tokens" in columns
    assert columns["cache_creation_input_tokens"].nullable is False
    assert columns["cache_read_input_tokens"].nullable is False


def test_model_pricing_table_has_cache_multiplier_columns() -> None:
    columns = Base.metadata.tables["model_pricing"].columns
    assert "cache_write_multiplier" in columns
    assert "cache_read_multiplier" in columns
    assert columns["cache_write_multiplier"].nullable is False
    assert columns["cache_read_multiplier"].nullable is False
