"""
Structured Screening UI Components for Streamlit

This module provides reusable UI components for structured screening:
1. render_chat_input() - Natural language input for requirements
2. render_requirements_editor() - Edit parsed requirements
3. render_screening_results() - Display screening results

Import into dashboard.py:
    from structured_screening_ui import (
        render_chat_input,
        render_requirements_editor,
        render_screening_results
    )
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from dataclasses import asdict
import json

# Import structured screening types
from structured_screening import (
    Requirement,
    RequirementType,
    CheckResult,
    ScreeningResult,
    parse_requirements,
)


# ============================================================================
# SESSION STATE KEYS
# ============================================================================

SS_PARSED_REQUIREMENTS = "structured_parsed_requirements"
SS_CHAT_INPUT = "structured_chat_input"
SS_PARSING_IN_PROGRESS = "structured_parsing_in_progress"
SS_PARSE_ERROR = "structured_parse_error"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _requirement_type_options() -> List[str]:
    """Get list of requirement type options for selectbox."""
    return [t.value for t in RequirementType]


def _get_requirement_type_label(req_type: RequirementType) -> str:
    """Get human-readable label for requirement type."""
    labels = {
        RequirementType.SKILL_FRONTEND: "Frontend Skill",
        RequirementType.SKILL_BACKEND: "Backend Skill",
        RequirementType.EXPERIENCE_YEARS: "Min Experience (Years)",
        RequirementType.LEADERSHIP_YEARS: "Leadership Experience",
        RequirementType.COMPANY_TYPE: "Company Type",
        RequirementType.TITLE_REJECT: "Reject Title Keywords",
        RequirementType.EXPERIENCE_MAX: "Max Experience (Years)",
        RequirementType.CUSTOM: "Custom Check",
    }
    return labels.get(req_type, req_type.value)


def _serialize_requirements(requirements: Dict[str, List[Requirement]]) -> Dict:
    """Serialize requirements to JSON-safe format for session state."""
    result = {}
    for section, reqs in requirements.items():
        result[section] = []
        for req in reqs:
            result[section].append({
                "type": req.type.value,
                "description": req.description,
                "values": req.values,
                "min_value": req.min_value,
                "max_value": req.max_value,
                "is_must_have": req.is_must_have,
                "boost_points": req.boost_points,
            })
    return result


def _deserialize_requirements(data: Dict) -> Dict[str, List[Requirement]]:
    """Deserialize requirements from session state."""
    result = {}
    for section, reqs in data.items():
        result[section] = []
        for req_data in reqs:
            result[section].append(Requirement(
                type=RequirementType(req_data["type"]),
                description=req_data["description"],
                values=req_data.get("values", []),
                min_value=req_data.get("min_value"),
                max_value=req_data.get("max_value"),
                is_must_have=req_data.get("is_must_have", True),
                boost_points=req_data.get("boost_points", 0),
            ))
    return result


def _create_empty_requirement(section: str) -> Requirement:
    """Create an empty requirement for a section."""
    if section == "must_have":
        return Requirement(
            type=RequirementType.CUSTOM,
            description="",
            values=[],
            is_must_have=True,
            boost_points=0,
        )
    elif section == "nice_to_have":
        return Requirement(
            type=RequirementType.CUSTOM,
            description="",
            values=[],
            is_must_have=False,
            boost_points=1,
        )
    else:  # reject_if
        return Requirement(
            type=RequirementType.COMPANY_TYPE,
            description="",
            values=[],
            is_must_have=True,
            boost_points=0,
        )


# ============================================================================
# CALLBACKS
# ============================================================================

def _cb_parse_requirements(anthropic_client):
    """Callback to parse requirements from chat input."""
    st.session_state[SS_PARSING_IN_PROGRESS] = True
    st.session_state[SS_PARSE_ERROR] = None

    try:
        input_text = st.session_state.get(SS_CHAT_INPUT, "")
        if not input_text.strip():
            st.session_state[SS_PARSE_ERROR] = "Please enter a job description or requirements."
            st.session_state[SS_PARSING_IN_PROGRESS] = False
            return

        # Parse requirements using AI
        requirements = parse_requirements(input_text, anthropic_client)

        # Store serialized requirements in session state
        st.session_state[SS_PARSED_REQUIREMENTS] = _serialize_requirements(requirements)
        st.session_state[SS_PARSING_IN_PROGRESS] = False

    except Exception as e:
        st.session_state[SS_PARSE_ERROR] = f"Error parsing requirements: {str(e)}"
        st.session_state[SS_PARSING_IN_PROGRESS] = False


def _cb_add_requirement(section: str):
    """Callback to add a new requirement to a section."""
    if SS_PARSED_REQUIREMENTS not in st.session_state:
        st.session_state[SS_PARSED_REQUIREMENTS] = {
            "must_have": [],
            "nice_to_have": [],
            "reject_if": [],
        }

    new_req = _create_empty_requirement(section)
    req_data = {
        "type": new_req.type.value,
        "description": new_req.description,
        "values": new_req.values,
        "min_value": new_req.min_value,
        "max_value": new_req.max_value,
        "is_must_have": new_req.is_must_have,
        "boost_points": new_req.boost_points,
    }
    st.session_state[SS_PARSED_REQUIREMENTS][section].append(req_data)


def _cb_remove_requirement(section: str, index: int):
    """Callback to remove a requirement from a section."""
    if SS_PARSED_REQUIREMENTS in st.session_state:
        reqs = st.session_state[SS_PARSED_REQUIREMENTS].get(section, [])
        if 0 <= index < len(reqs):
            reqs.pop(index)


def _cb_toggle_requirement(section: str, index: int, enabled: bool):
    """Callback to enable/disable a requirement."""
    if SS_PARSED_REQUIREMENTS in st.session_state:
        reqs = st.session_state[SS_PARSED_REQUIREMENTS].get(section, [])
        if 0 <= index < len(reqs):
            # For must_have and reject_if, toggling disabled removes from consideration
            # We'll use a special "_enabled" field to track this
            reqs[index]["_enabled"] = enabled


def _cb_clear_requirements():
    """Callback to clear all parsed requirements."""
    st.session_state.pop(SS_PARSED_REQUIREMENTS, None)
    st.session_state.pop(SS_PARSE_ERROR, None)


# ============================================================================
# COMPONENT 1: CHAT INPUT
# ============================================================================

def render_chat_input(anthropic_client=None, on_parse: Optional[Callable] = None) -> Optional[str]:
    """
    Render a text area for natural language requirement input.

    Args:
        anthropic_client: Anthropic client for parsing (optional, shows parse button if provided)
        on_parse: Optional callback when parsing completes

    Returns:
        The current input text, or None if empty
    """
    st.subheader("Describe What You're Looking For")

    # Help text
    st.caption(
        "Describe your ideal candidate in natural language. Include required skills, "
        "experience levels, company types to target or avoid, and any other criteria."
    )

    # Example placeholder
    placeholder_text = """Example:
Looking for a Fullstack Team Lead with:
- 5+ years of fullstack experience
- 2+ years leading a team
- Must have React and Node.js
- Currently at a software product company (not consulting)
- Reject if title is pure Backend or Frontend
- Bonus: Top company (Wiz, Monday, Snyk) or elite unit (8200, Mamram)"""

    # Text area for input
    input_text = st.text_area(
        "Job Description / Requirements",
        key=SS_CHAT_INPUT,
        height=200,
        placeholder=placeholder_text,
        label_visibility="collapsed",
    )

    # Parse button and status
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        parse_disabled = not anthropic_client or st.session_state.get(SS_PARSING_IN_PROGRESS, False)

        if st.button(
            "Parse Requirements",
            type="primary",
            disabled=parse_disabled,
            use_container_width=True,
        ):
            if anthropic_client:
                _cb_parse_requirements(anthropic_client)
                if on_parse:
                    on_parse()

    with col2:
        if st.button(
            "Clear All",
            disabled=st.session_state.get(SS_PARSING_IN_PROGRESS, False),
            use_container_width=True,
        ):
            _cb_clear_requirements()
            st.session_state[SS_CHAT_INPUT] = ""

    # Show parsing status
    if st.session_state.get(SS_PARSING_IN_PROGRESS, False):
        st.info("Parsing requirements...")

    # Show parse error if any
    error = st.session_state.get(SS_PARSE_ERROR)
    if error:
        st.error(error)

    # Show success message when requirements are parsed
    if SS_PARSED_REQUIREMENTS in st.session_state and not error:
        reqs = st.session_state[SS_PARSED_REQUIREMENTS]
        total = sum(len(reqs.get(s, [])) for s in ["must_have", "nice_to_have", "reject_if"])
        st.success(f"Parsed {total} requirements. Review and edit below.")

    return input_text if input_text else None


# ============================================================================
# COMPONENT 2: REQUIREMENTS EDITOR
# ============================================================================

def _render_single_requirement(
    req_data: Dict,
    section: str,
    index: int,
    show_boost: bool = False,
    show_checkbox: bool = True,
) -> Dict:
    """Render a single requirement editor row and return updated data."""

    # Create a unique key prefix for this requirement
    key_prefix = f"{section}_{index}"

    # Check if enabled (default True)
    enabled = req_data.get("_enabled", True)

    # Create columns: checkbox | type | description | values/numbers | remove
    if show_checkbox:
        cols = st.columns([0.5, 1.5, 2.5, 2.5, 0.5])
        col_idx = 0

        with cols[col_idx]:
            new_enabled = st.checkbox(
                "Enable",
                value=enabled,
                key=f"{key_prefix}_enabled",
                label_visibility="collapsed",
            )
            req_data["_enabled"] = new_enabled
        col_idx += 1
    else:
        cols = st.columns([1.5, 2.5, 2.5, 0.5])
        col_idx = 0

    # Type selector
    with cols[col_idx]:
        current_type = req_data.get("type", "custom")
        type_options = _requirement_type_options()
        type_index = type_options.index(current_type) if current_type in type_options else 0

        new_type = st.selectbox(
            "Type",
            options=type_options,
            index=type_index,
            key=f"{key_prefix}_type",
            label_visibility="collapsed",
            disabled=not enabled if show_checkbox else False,
        )
        req_data["type"] = new_type
    col_idx += 1

    # Description
    with cols[col_idx]:
        new_desc = st.text_input(
            "Description",
            value=req_data.get("description", ""),
            key=f"{key_prefix}_desc",
            label_visibility="collapsed",
            disabled=not enabled if show_checkbox else False,
        )
        req_data["description"] = new_desc
    col_idx += 1

    # Values / Min/Max based on type
    with cols[col_idx]:
        req_type = RequirementType(new_type)

        if req_type in [RequirementType.EXPERIENCE_YEARS, RequirementType.LEADERSHIP_YEARS]:
            # Numeric min value
            min_val = st.number_input(
                "Min Years",
                min_value=0,
                max_value=50,
                value=int(req_data.get("min_value") or 0),
                key=f"{key_prefix}_min",
                label_visibility="collapsed",
                disabled=not enabled if show_checkbox else False,
            )
            req_data["min_value"] = min_val
            req_data["values"] = []

        elif req_type == RequirementType.EXPERIENCE_MAX:
            # Numeric max value
            max_val = st.number_input(
                "Max Years",
                min_value=0,
                max_value=50,
                value=int(req_data.get("max_value") or 20),
                key=f"{key_prefix}_max",
                label_visibility="collapsed",
                disabled=not enabled if show_checkbox else False,
            )
            req_data["max_value"] = max_val
            req_data["values"] = []

        else:
            # Text values (comma-separated)
            current_values = req_data.get("values", [])
            values_str = ", ".join(current_values) if current_values else ""

            new_values_str = st.text_input(
                "Values (comma-separated)",
                value=values_str,
                key=f"{key_prefix}_values",
                placeholder="React, Vue, Angular",
                label_visibility="collapsed",
                disabled=not enabled if show_checkbox else False,
            )

            # Parse comma-separated values
            if new_values_str:
                req_data["values"] = [v.strip() for v in new_values_str.split(",") if v.strip()]
            else:
                req_data["values"] = []
    col_idx += 1

    # Remove button (or boost points for nice_to_have)
    with cols[col_idx]:
        if show_boost:
            # Show boost points input instead of remove for nice_to_have
            boost = st.number_input(
                "Boost",
                min_value=0,
                max_value=5,
                value=req_data.get("boost_points", 1),
                key=f"{key_prefix}_boost",
                label_visibility="collapsed",
            )
            req_data["boost_points"] = boost
        else:
            if st.button("X", key=f"{key_prefix}_remove", help="Remove this requirement"):
                return None  # Signal to remove

    return req_data


def render_requirements_editor() -> Optional[Dict[str, List[Requirement]]]:
    """
    Render the requirements editor UI.

    Shows parsed requirements in editable boxes organized by section:
    - Must-Have Requirements (with checkboxes to enable/disable)
    - Nice-to-Have (Boosters) (with boost point inputs)
    - Reject If (with checkboxes)

    Returns:
        Dictionary of requirements by section, or None if no requirements parsed
    """

    if SS_PARSED_REQUIREMENTS not in st.session_state:
        st.info("Parse a job description above, or manually add requirements.")

        # Show add buttons even without parsed requirements
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("+ Add Must-Have", use_container_width=True):
                st.session_state[SS_PARSED_REQUIREMENTS] = {
                    "must_have": [],
                    "nice_to_have": [],
                    "reject_if": [],
                }
                _cb_add_requirement("must_have")
                st.rerun()
        with col2:
            if st.button("+ Add Nice-to-Have", use_container_width=True):
                st.session_state[SS_PARSED_REQUIREMENTS] = {
                    "must_have": [],
                    "nice_to_have": [],
                    "reject_if": [],
                }
                _cb_add_requirement("nice_to_have")
                st.rerun()
        with col3:
            if st.button("+ Add Reject-If", use_container_width=True):
                st.session_state[SS_PARSED_REQUIREMENTS] = {
                    "must_have": [],
                    "nice_to_have": [],
                    "reject_if": [],
                }
                _cb_add_requirement("reject_if")
                st.rerun()
        return None

    requirements_data = st.session_state[SS_PARSED_REQUIREMENTS]

    # Track which requirements to remove
    to_remove = []

    # Section 1: Must-Have Requirements
    st.markdown("### Must-Have Requirements")
    st.caption("All enabled requirements must be met for a candidate to pass.")

    must_have = requirements_data.get("must_have", [])
    if must_have:
        # Header row
        if len(must_have) > 0:
            header_cols = st.columns([0.5, 1.5, 2.5, 2.5, 0.5])
            header_cols[0].markdown("**On**")
            header_cols[1].markdown("**Type**")
            header_cols[2].markdown("**Description**")
            header_cols[3].markdown("**Values / Threshold**")
            header_cols[4].markdown("")

        for i, req_data in enumerate(must_have):
            updated = _render_single_requirement(
                req_data, "must_have", i, show_boost=False, show_checkbox=True
            )
            if updated is None:
                to_remove.append(("must_have", i))
    else:
        st.caption("No must-have requirements defined.")

    if st.button("+ Add Must-Have Requirement", key="add_must_have"):
        _cb_add_requirement("must_have")
        st.rerun()

    st.divider()

    # Section 2: Nice-to-Have (Boosters)
    st.markdown("### Nice-to-Have (Boosters)")
    st.caption("Bonus points added to score when these are met. Base score is 6 for passing all must-haves.")

    nice_to_have = requirements_data.get("nice_to_have", [])
    if nice_to_have:
        # Header row
        header_cols = st.columns([1.5, 2.5, 2.5, 0.5])
        header_cols[0].markdown("**Type**")
        header_cols[1].markdown("**Description**")
        header_cols[2].markdown("**Values**")
        header_cols[3].markdown("**+Pts**")

        for i, req_data in enumerate(nice_to_have):
            # For nice-to-have, show boost points column instead of remove
            key_prefix = f"nice_to_have_{i}"

            cols = st.columns([1.5, 2.5, 2.5, 0.5])

            # Type
            with cols[0]:
                current_type = req_data.get("type", "custom")
                type_options = _requirement_type_options()
                type_index = type_options.index(current_type) if current_type in type_options else 0
                new_type = st.selectbox(
                    "Type", options=type_options, index=type_index,
                    key=f"{key_prefix}_type", label_visibility="collapsed"
                )
                req_data["type"] = new_type

            # Description
            with cols[1]:
                new_desc = st.text_input(
                    "Description", value=req_data.get("description", ""),
                    key=f"{key_prefix}_desc", label_visibility="collapsed"
                )
                req_data["description"] = new_desc

            # Values
            with cols[2]:
                current_values = req_data.get("values", [])
                values_str = ", ".join(current_values) if current_values else ""
                new_values_str = st.text_input(
                    "Values", value=values_str,
                    key=f"{key_prefix}_values", label_visibility="collapsed"
                )
                if new_values_str:
                    req_data["values"] = [v.strip() for v in new_values_str.split(",") if v.strip()]
                else:
                    req_data["values"] = []

            # Boost points
            with cols[3]:
                boost = st.number_input(
                    "Boost", min_value=0, max_value=5,
                    value=req_data.get("boost_points", 1),
                    key=f"{key_prefix}_boost", label_visibility="collapsed"
                )
                req_data["boost_points"] = boost
    else:
        st.caption("No nice-to-have requirements defined.")

    col_add, col_remove = st.columns([1, 1])
    with col_add:
        if st.button("+ Add Nice-to-Have", key="add_nice_to_have"):
            _cb_add_requirement("nice_to_have")
            st.rerun()
    with col_remove:
        if nice_to_have and st.button("Remove Last", key="remove_last_nice"):
            requirements_data["nice_to_have"].pop()
            st.rerun()

    st.divider()

    # Section 3: Reject If
    st.markdown("### Reject If")
    st.caption("Candidates matching any of these criteria will be rejected.")

    reject_if = requirements_data.get("reject_if", [])
    if reject_if:
        # Header row
        header_cols = st.columns([0.5, 1.5, 2.5, 2.5, 0.5])
        header_cols[0].markdown("**On**")
        header_cols[1].markdown("**Type**")
        header_cols[2].markdown("**Description**")
        header_cols[3].markdown("**Values / Keywords**")
        header_cols[4].markdown("")

        for i, req_data in enumerate(reject_if):
            updated = _render_single_requirement(
                req_data, "reject_if", i, show_boost=False, show_checkbox=True
            )
            if updated is None:
                to_remove.append(("reject_if", i))
    else:
        st.caption("No reject-if rules defined.")

    if st.button("+ Add Reject-If Rule", key="add_reject_if"):
        _cb_add_requirement("reject_if")
        st.rerun()

    # Process removals (in reverse order to maintain indices)
    if to_remove:
        for section, idx in reversed(to_remove):
            requirements_data[section].pop(idx)
        st.rerun()

    # Return deserialized requirements
    return _deserialize_requirements(requirements_data)


# ============================================================================
# COMPONENT 3: SCREENING RESULTS
# ============================================================================

def render_screening_results(results: List[ScreeningResult]) -> None:
    """
    Render screening results with color coding and expandable details.

    Args:
        results: List of ScreeningResult objects from screening
    """
    if not results:
        st.info("No screening results to display.")
        return

    st.subheader(f"Screening Results ({len(results)} profiles)")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    strong_fit = sum(1 for r in results if r.fit == "Strong Fit")
    good_fit = sum(1 for r in results if r.fit == "Good Fit")
    partial_fit = sum(1 for r in results if r.fit == "Partial Fit")
    not_fit = sum(1 for r in results if r.fit == "Not a Fit")

    col1.metric("Strong Fit", strong_fit)
    col2.metric("Good Fit", good_fit)
    col3.metric("Partial Fit", partial_fit)
    col4.metric("Not a Fit", not_fit)

    st.divider()

    # Build results dataframe
    df_data = []
    for r in results:
        # Count passed/failed checks
        must_have_checks = [c for c in r.checks if c.requirement.is_must_have]
        failed_checks = [c for c in must_have_checks if not c.passed]
        passed_checks = [c for c in must_have_checks if c.passed]

        # Format failed/passed as strings
        failed_str = "; ".join([c.reason for c in failed_checks]) if failed_checks else "-"
        passed_str = "; ".join([c.reason for c in passed_checks[:3]]) if passed_checks else "-"

        df_data.append({
            "Name": r.profile_name,
            "Score": r.score,
            "Fit": r.fit,
            "Failed Checks": failed_str,
            "Passed Checks": passed_str,
            "_result": r,  # Store full result for expansion
        })

    df = pd.DataFrame(df_data)

    # Sort by score descending
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)

    # Color coding function
    def get_fit_color(fit: str) -> str:
        colors = {
            "Strong Fit": "#22c55e",   # Green
            "Good Fit": "#84cc16",     # Lime
            "Partial Fit": "#eab308",  # Yellow
            "Not a Fit": "#ef4444",    # Red
            "Error": "#9ca3af",        # Gray
        }
        return colors.get(fit, "#9ca3af")

    # Render each result row with expandable details
    for idx, row in df.iterrows():
        result: ScreeningResult = row["_result"]
        fit_color = get_fit_color(result.fit)

        # Create a colored container for each profile
        with st.container():
            # Header row with basic info
            cols = st.columns([3, 1, 1.5, 4])

            with cols[0]:
                st.markdown(f"**{result.profile_name}**")

            with cols[1]:
                st.markdown(f"**Score: {result.score}/10**")

            with cols[2]:
                # Colored fit badge
                st.markdown(
                    f'<span style="background-color: {fit_color}; color: white; '
                    f'padding: 2px 8px; border-radius: 4px; font-weight: 600;">'
                    f'{result.fit}</span>',
                    unsafe_allow_html=True
                )

            with cols[3]:
                # Summary text
                st.caption(result.summary[:100] + "..." if len(result.summary) > 100 else result.summary)

            # Expandable details
            with st.expander("View Details", expanded=False):
                # Organize checks by section
                must_have_checks = [c for c in result.checks if c.requirement.is_must_have]
                nice_to_have_checks = [c for c in result.checks if not c.requirement.is_must_have]

                # Must-have results
                st.markdown("**Must-Have Checks:**")
                for check in must_have_checks:
                    icon = "PASS" if check.passed else "FAIL"
                    icon_color = "#22c55e" if check.passed else "#ef4444"

                    st.markdown(
                        f'<span style="color: {icon_color}; font-weight: bold;">[{icon}]</span> '
                        f'{_get_requirement_type_label(check.requirement.type)}: {check.reason}',
                        unsafe_allow_html=True
                    )

                    if check.evidence:
                        evidence_str = ", ".join(str(e)[:50] for e in check.evidence[:3])
                        st.caption(f"Evidence: {evidence_str}")

                # Nice-to-have results (if any passed)
                passed_nice = [c for c in nice_to_have_checks if c.passed]
                if passed_nice:
                    st.markdown("**Bonus Points:**")
                    for check in passed_nice:
                        st.markdown(
                            f'<span style="color: #22c55e;">+{check.requirement.boost_points}</span> '
                            f'{check.requirement.description}: {check.reason}',
                            unsafe_allow_html=True
                        )

            st.divider()

    # Export button
    st.markdown("### Export Results")

    # Create export dataframe (without internal _result column)
    export_df = df.drop(columns=["_result"])

    csv = export_df.to_csv(index=False).encode('utf-8-sig')

    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="screening_results.csv",
        mime="text/csv",
    )


# ============================================================================
# CONVENIENCE: FULL SCREENING UI
# ============================================================================

def render_structured_screening_tab(
    anthropic_client=None,
    profiles: Optional[List[Dict]] = None,
    on_screen: Optional[Callable] = None,
):
    """
    Render a complete structured screening tab.

    This combines all three components into a cohesive UI flow:
    1. Chat input for requirements
    2. Requirements editor
    3. Screen button
    4. Results display

    Args:
        anthropic_client: Anthropic client for AI operations
        profiles: List of profiles to screen (if available)
        on_screen: Callback when screening is requested, receives requirements dict
    """
    st.header("Structured Screening")

    # Step 1: Chat input
    render_chat_input(anthropic_client)

    st.divider()

    # Step 2: Requirements editor
    st.subheader("Review & Edit Requirements")
    requirements = render_requirements_editor()

    # Step 3: Screen button (if we have both requirements and profiles)
    if requirements and profiles:
        st.divider()

        total_reqs = sum(len(requirements.get(s, [])) for s in ["must_have", "nice_to_have", "reject_if"])
        enabled_must_have = len([
            r for r in requirements.get("must_have", [])
            if getattr(r, '_enabled', True)
        ])

        st.info(f"Ready to screen **{len(profiles)}** profiles against **{total_reqs}** requirements.")

        if st.button("Run Structured Screening", type="primary"):
            if on_screen:
                on_screen(requirements)
    elif requirements and not profiles:
        st.warning("No profiles loaded. Please upload profiles in the main screening tab.")

    # Step 4: Show results if available
    if "structured_screening_results" in st.session_state:
        results = st.session_state["structured_screening_results"]
        render_screening_results(results)


# ============================================================================
# UTILITIES FOR INTEGRATION
# ============================================================================

def get_current_requirements() -> Optional[Dict[str, List[Requirement]]]:
    """
    Get the current requirements from session state.

    Returns:
        Dictionary of requirements by section, or None if not set
    """
    if SS_PARSED_REQUIREMENTS not in st.session_state:
        return None
    return _deserialize_requirements(st.session_state[SS_PARSED_REQUIREMENTS])


def set_requirements(requirements: Dict[str, List[Requirement]]) -> None:
    """
    Set requirements in session state programmatically.

    Args:
        requirements: Dictionary of requirements by section
    """
    st.session_state[SS_PARSED_REQUIREMENTS] = _serialize_requirements(requirements)


def clear_requirements() -> None:
    """Clear all requirements from session state."""
    _cb_clear_requirements()
