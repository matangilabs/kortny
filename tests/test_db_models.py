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
        "dashboard_users",
        "dashboard_oauth_states",
        "composio_connections",
        "observe_policies",
        "observation_events",
        "observe_channel_profiles",
        "procedural_skills",
        "procedural_skill_versions",
        "procedural_skill_invocations",
        "episodes",
        "llm_usage",
        "artifacts",
        "model_pricing",
    }


def test_task_status_enum_matches_locked_schema() -> None:
    assert [status.value for status in TaskStatus] == [
        "pending",
        "running",
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
