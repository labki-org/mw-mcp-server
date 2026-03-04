"""
SMW Query Validation Tests

Tests for property, category, and printout validation in tool_run_smw_ask,
including the _find_best_match fuzzy-matching helper.
"""

import pytest
from unittest.mock import AsyncMock

from mw_mcp_server.auth.models import UserContext
from mw_mcp_server.tools.wiki_tools import _find_best_match, tool_run_smw_ask


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

KNOWN_PROPS = {
    "Property:Has center role",
    "Property:Population",
    "Property:Has name",
    "Property:Located in",
}

KNOWN_CATS = {
    "Category:City",
    "Category:Town",
    "Category:Director",
    "Category:Person",
}


@pytest.fixture
def props_lower():
    return {p.lower(): p for p in KNOWN_PROPS}


@pytest.fixture
def cats_lower():
    return {c.lower(): c for c in KNOWN_CATS}


@pytest.fixture
def user():
    return UserContext(
        username="TestUser",
        wiki_id="test-wiki",
        user_id=1,
        client_id="test",
        allowed_namespaces=[0, 14, 102],
    )


@pytest.fixture
def mock_vector_store():
    """VectorStore mock whose get_pages_by_namespace returns props or cats."""
    store = AsyncMock()

    async def _get_pages(wiki_id, ns, pattern=None):
        if ns == 102:
            return list(KNOWN_PROPS)
        if ns == 14:
            return list(KNOWN_CATS)
        return []

    store.get_pages_by_namespace = AsyncMock(side_effect=_get_pages)
    return store


@pytest.fixture
def mock_smw_client():
    client = AsyncMock()
    client.ask = AsyncMock(return_value={"results": {}})
    return client


# ===================================================================
# _find_best_match — unit tests
# ===================================================================


class TestFindBestMatch:
    """Tests for the _find_best_match helper."""

    def test_exact_match_passes(self, props_lower):
        _find_best_match("Has center role", KNOWN_PROPS, props_lower, "Property:")

    def test_exact_category_passes(self, cats_lower):
        _find_best_match("City", KNOWN_CATS, cats_lower, "Category:")

    def test_has_prefix_suggestion(self, props_lower):
        with pytest.raises(ValueError, match="Has center role"):
            _find_best_match("center role", KNOWN_PROPS, props_lower, "Property:")

    def test_has_prefix_not_applied_to_categories(self, cats_lower):
        # "Has City" doesn't exist and Category namespace shouldn't try "Has " prefix
        _find_best_match("Has City", KNOWN_CATS, cats_lower, "Category:")

    def test_case_mismatch_property(self, props_lower):
        with pytest.raises(ValueError, match="Case Mismatch"):
            _find_best_match("has center role", KNOWN_PROPS, props_lower, "Property:")

    def test_case_mismatch_category(self, cats_lower):
        with pytest.raises(ValueError, match="Case Mismatch"):
            _find_best_match("city", KNOWN_CATS, cats_lower, "Category:")

    def test_plural_to_singular_suggestion(self, props_lower):
        with pytest.raises(ValueError, match="Property:Population"):
            _find_best_match("Populations", KNOWN_PROPS, props_lower, "Property:")

    def test_singular_to_plural_suggestion(self):
        """'Director' exists; searching 'Director' with an extra 's' set should suggest it."""
        known = {"Category:Directors"}
        lower = {c.lower(): c for c in known}
        with pytest.raises(ValueError, match="Category:Directors"):
            _find_best_match("Director", known, lower, "Category:")

    def test_unknown_name_passes_silently(self, props_lower):
        _find_best_match("CompletelyUnknown", KNOWN_PROPS, props_lower, "Property:")

    def test_unknown_category_passes_silently(self, cats_lower):
        _find_best_match("Nonexistent", KNOWN_CATS, cats_lower, "Category:")

    def test_tool_hint_properties(self, props_lower):
        with pytest.raises(ValueError, match="mw_get_properties"):
            _find_best_match("center role", KNOWN_PROPS, props_lower, "Property:")

    def test_tool_hint_categories(self):
        """Tool hint appears in singular/plural suggestion for categories."""
        known = {"Category:Directors"}
        lower = {c.lower(): c for c in known}
        with pytest.raises(ValueError, match="mw_get_categories"):
            _find_best_match("Director", known, lower, "Category:")


# ===================================================================
# tool_run_smw_ask — integration tests
# ===================================================================


class TestSmwAskValidation:
    """Tests for the validation logic inside tool_run_smw_ask."""

    async def test_valid_property_condition(self, user, mock_vector_store, mock_smw_client):
        await tool_run_smw_ask(
            "[[Has center role::Director]]",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        mock_smw_client.ask.assert_called_once()

    async def test_misspelled_property_raises(self, user, mock_vector_store, mock_smw_client):
        with pytest.raises(ValueError, match="Has center role"):
            await tool_run_smw_ask(
                "[[center role::Director]]",
                user,
                client=mock_smw_client,
                vector_store=mock_vector_store,
            )

    async def test_valid_category_condition(self, user, mock_vector_store, mock_smw_client):
        await tool_run_smw_ask(
            "[[Category:City]]",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        mock_smw_client.ask.assert_called_once()

    async def test_misspelled_category_raises(self, user, mock_vector_store, mock_smw_client):
        with pytest.raises(ValueError, match="Case Mismatch"):
            await tool_run_smw_ask(
                "[[Category:city]]",
                user,
                client=mock_smw_client,
                vector_store=mock_vector_store,
            )

    async def test_or_categories_both_validated(self, user, mock_vector_store, mock_smw_client):
        await tool_run_smw_ask(
            "[[Category:City||Category:Town]]",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        mock_smw_client.ask.assert_called_once()

    async def test_or_categories_second_misspelled(self, user, mock_vector_store, mock_smw_client):
        with pytest.raises(ValueError, match="Case Mismatch"):
            await tool_run_smw_ask(
                "[[Category:City||Category:town]]",
                user,
                client=mock_smw_client,
                vector_store=mock_vector_store,
            )

    async def test_valid_printout_property(self, user, mock_vector_store, mock_smw_client):
        await tool_run_smw_ask(
            "[[Has center role::Director]]|?Population",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        mock_smw_client.ask.assert_called_once()

    async def test_misspelled_printout_raises(self, user, mock_vector_store, mock_smw_client):
        with pytest.raises(ValueError, match="Has center role"):
            await tool_run_smw_ask(
                "[[Category:Director]]|?center role",
                user,
                client=mock_smw_client,
                vector_store=mock_vector_store,
            )

    async def test_printout_with_label(self, user, mock_vector_store, mock_smw_client):
        await tool_run_smw_ask(
            "[[Category:City]]|?Population=Pop",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        mock_smw_client.ask.assert_called_once()

    async def test_printout_with_format(self, user, mock_vector_store, mock_smw_client):
        await tool_run_smw_ask(
            "[[Category:City]]|?Population#-",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        mock_smw_client.ask.assert_called_once()

    async def test_special_printout_category_skipped(self, user, mock_vector_store, mock_smw_client):
        """'|?Category' is a built-in SMW printout, not a real property."""
        await tool_run_smw_ask(
            "[[Has center role::Director]]|?Category",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        mock_smw_client.ask.assert_called_once()

    async def test_special_printout_mainlabel_skipped(self, user, mock_vector_store, mock_smw_client):
        await tool_run_smw_ask(
            "[[Category:City]]|?Mainlabel=Name",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        mock_smw_client.ask.assert_called_once()

    async def test_no_vector_store_skips_validation(self, user, mock_smw_client):
        """With vector_store=None, all validation is skipped."""
        await tool_run_smw_ask(
            "[[Nonexistent property::X]]|?Also fake",
            user,
            client=mock_smw_client,
            vector_store=None,
        )
        mock_smw_client.ask.assert_called_once()

    async def test_empty_query_raises(self, user):
        with pytest.raises(ValueError, match="non-empty"):
            await tool_run_smw_ask("", user)

    async def test_no_namespace_access_returns_empty(self, mock_smw_client, mock_vector_store):
        no_access_user = UserContext(
            username="Restricted",
            wiki_id="test-wiki",
            user_id=2,
            client_id="test",
            allowed_namespaces=[],
        )
        result = await tool_run_smw_ask(
            "[[Has center role::Director]]",
            no_access_user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        assert result == {"result": "", "filtered_count": 0}
        mock_smw_client.ask.assert_not_called()

    async def test_combined_query_all_valid(self, user, mock_vector_store, mock_smw_client):
        """Full query with conditions, categories, and printouts."""
        await tool_run_smw_ask(
            "[[Has center role::Director]][[Category:Person]]|?Has name|?Population",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        mock_smw_client.ask.assert_called_once()

    async def test_combined_query_bad_printout(self, user, mock_vector_store, mock_smw_client):
        with pytest.raises(ValueError, match="Has name"):
            await tool_run_smw_ask(
                "[[Has center role::Director]][[Category:Person]]|?name",
                user,
                client=mock_smw_client,
                vector_store=mock_vector_store,
            )


# ===================================================================
# Concurrency — verify gather is used for both namespaces
# ===================================================================


class TestSmwAskConcurrency:
    """Verify that DB calls are made only for needed namespaces."""

    async def test_only_props_fetched_when_no_categories(
        self, user, mock_vector_store, mock_smw_client
    ):
        await tool_run_smw_ask(
            "[[Has center role::Director]]|?Population",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        # Only NS_PROPERTY should be fetched (no categories in query)
        calls = mock_vector_store.get_pages_by_namespace.call_args_list
        namespaces_fetched = {c.args[1] for c in calls}
        assert namespaces_fetched == {102}

    async def test_only_cats_fetched_when_no_properties(
        self, user, mock_vector_store, mock_smw_client
    ):
        await tool_run_smw_ask(
            "[[Category:City]]",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        calls = mock_vector_store.get_pages_by_namespace.call_args_list
        namespaces_fetched = {c.args[1] for c in calls}
        assert namespaces_fetched == {14}

    async def test_both_fetched_for_mixed_query(
        self, user, mock_vector_store, mock_smw_client
    ):
        await tool_run_smw_ask(
            "[[Has center role::Director]][[Category:Person]]|?Population",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        calls = mock_vector_store.get_pages_by_namespace.call_args_list
        namespaces_fetched = {c.args[1] for c in calls}
        assert namespaces_fetched == {14, 102}

    async def test_no_db_calls_when_query_has_no_refs(
        self, user, mock_vector_store, mock_smw_client
    ):
        """A plain query with no properties or categories skips DB entirely."""
        await tool_run_smw_ask(
            "[[+]]",
            user,
            client=mock_smw_client,
            vector_store=mock_vector_store,
        )
        mock_vector_store.get_pages_by_namespace.assert_not_called()
