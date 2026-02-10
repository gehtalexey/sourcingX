"""Quick test for screening fixes. Run: python test_screening_fixes.py"""
import json

# ============================================================
# Test 1: format_past_positions includes dates/durations
# ============================================================
print("=" * 60)
print("TEST 1: format_past_positions with rich data")
print("=" * 60)
from helpers import format_past_positions, format_education

positions = [
    {'employee_title': 'Senior Backend Engineer', 'employer_name': 'Wiz', 'start_date': '2021-03', 'end_date': '2023-12', 'duration': '2y 9m'},
    {'employee_title': 'Software Engineer', 'employer_name': 'Google', 'start_date': '2018-06', 'end_date': '2021-02'},
    {'employee_title': 'Junior Developer', 'employer_name': 'Matrix'},
]
result = format_past_positions(positions)
print(f"  Result: {result}")
assert 'Wiz' in result and '2021-03' in result and '2y 9m' in result, "Missing date/duration info!"
assert 'Google' in result and 'Matrix' in result, "Missing companies!"
print("  PASSED\n")

# ============================================================
# Test 2: format_education with structured data
# ============================================================
print("=" * 60)
print("TEST 2: format_education with degree/field")
print("=" * 60)
edu = [
    {'school_name': 'Technion', 'degree': 'BSc', 'field_of_study': 'Computer Science'},
    {'school_name': 'Tel Aviv University'},
]
result = format_education(edu)
print(f"  Result: {result}")
assert 'BSc in Computer Science' in result, "Missing degree info!"
assert 'Tel Aviv University' in result, "Missing school!"
print("  PASSED\n")

# ============================================================
# Test 3: screen_profile extracts from raw_crustdata
# ============================================================
print("=" * 60)
print("TEST 3: screen_profile builds rich profile from raw_crustdata")
print("=" * 60)

# Build a mock profile like what enriched_df.to_dict('records') produces
mock_profile = {
    'first_name': 'Test',
    'last_name': 'User',
    'current_title': 'Senior Engineer',
    'current_company': 'Wiz',
    'headline': 'Senior Engineer at Wiz',
    'location': 'Tel Aviv',
    'summary': 'Experienced backend developer',
    'skills': 'Python, Node.js, AWS, Kubernetes',
    'education': 'Technion',
    'past_positions': 'Software Engineer at Google | Junior Dev at Startup',
    'raw_crustdata': {
        'past_employers': [
            {'employee_title': 'Software Engineer', 'employer_name': 'Google', 'start_date': '2018-06', 'end_date': '2021-02'},
            {'employee_title': 'Junior Developer', 'employer_name': 'Startup XYZ', 'start_date': '2016-01', 'end_date': '2018-05'},
        ],
        'current_employers': [
            {'employee_title': 'Senior Engineer', 'employer_name': 'Wiz', 'start_date': '2021-03'},
        ],
        'education_background': [
            {'school_name': 'Technion', 'degree': 'BSc', 'field_of_study': 'Computer Science'},
        ],
        'all_employers': ['Wiz', 'Google', 'Startup XYZ'],
        'all_titles': ['Senior Engineer', 'Software Engineer', 'Junior Developer'],
    }
}

# Import screen_profile and patch it to capture the prompt without calling OpenAI
import dashboard

captured_prompt = {}

class MockResponse:
    class Usage:
        prompt_tokens = 100
        completion_tokens = 50
    class Choice:
        class Message:
            content = json.dumps({
                "score": 8, "fit": "Strong Fit",
                "summary": "Test candidate",
                "why": "Good match",
                "strengths": ["Python", "AWS"],
                "concerns": ["Junior background"]
            })
        message = Message()
    choices = [Choice()]
    usage = Usage()

class MockClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                captured_prompt['messages'] = kwargs['messages']
                captured_prompt['response_format'] = kwargs.get('response_format')
                return MockResponse()

result = dashboard.screen_profile(mock_profile, "Backend engineer with 5+ years", MockClient())

# Check the prompt that was built
user_msg = captured_prompt['messages'][1]['content']
print(f"  Profile section sent to AI:")
for line in user_msg.split('\n'):
    if any(k in line for k in ['Name:', 'Title:', 'Education:', 'Past Positions:', 'Current Position:', 'All Employers:', 'All Titles:']):
        print(f"    {line}")

# Verify rich data was extracted
assert 'BSc in Computer Science' in user_msg, "Education degree not extracted from raw_crustdata!"
assert 'Google' in user_msg and '2018-06' in user_msg, "Past positions dates not extracted!"
assert 'Wiz, Google, Startup XYZ' in user_msg, "All employers not included!"
assert 'Senior Engineer, Software Engineer' in user_msg, "All titles not included!"
print("  PASSED\n")

# ============================================================
# Test 4: response_format=json_object is set
# ============================================================
print("=" * 60)
print("TEST 4: response_format=json_object is set")
print("=" * 60)
assert captured_prompt['response_format'] == {"type": "json_object"}, f"Expected json_object, got: {captured_prompt['response_format']}"
print("  PASSED\n")

# ============================================================
# Test 5: AI returns 'why' field, verify it's in result
# ============================================================
print("=" * 60)
print("TEST 5: Result contains 'why' field (not 'reasoning')")
print("=" * 60)
assert result.get('why') == 'Good match', f"Expected 'why' field, got: {result}"
assert 'reasoning' not in result, "Should not have 'reasoning' field!"
print(f"  Result: score={result['score']}, fit={result['fit']}, why={result['why']}")
print("  PASSED\n")

# ============================================================
# Test 6: API key validation uses models.list (not chat)
# ============================================================
print("=" * 60)
print("TEST 6: API key validation uses models.list()")
print("=" * 60)
import inspect
source = inspect.getsource(dashboard)
# Check that the old "Say OK" pattern is gone
assert '"Say OK"' not in source, "Old 'Say OK' validation still present!"
assert 'models.list()' in source, "models.list() validation not found!"
print("  PASSED\n")

# ============================================================
# Test 7: Skipped profile (no data)
# ============================================================
print("=" * 60)
print("TEST 7: Empty profile is skipped")
print("=" * 60)
empty_result = dashboard.screen_profile({'first_name': '', 'last_name': ''}, "job desc", MockClient())
assert empty_result['fit'] == 'Skipped', f"Expected Skipped, got: {empty_result['fit']}"
print(f"  Result: {empty_result['summary']}")
print("  PASSED\n")

# ============================================================
# Test 8: Fallback when no raw_crustdata (DB path)
# ============================================================
print("=" * 60)
print("TEST 8: Fallback to flat fields when no raw_crustdata")
print("=" * 60)
db_profile = {
    'first_name': 'Jane',
    'last_name': 'Doe',
    'current_title': 'Developer',
    'current_company': 'Startup',
    'location': 'NYC',
    'skills': 'React, TypeScript',
    'summary': 'Full-stack developer',
    'education': 'MIT',
    'past_positions': 'Engineer at Facebook | Intern at Amazon',
    'all_employers': 'Facebook, Amazon, Startup',
    'all_titles': 'Engineer, Intern, Developer',
}
captured_prompt.clear()
result2 = dashboard.screen_profile(db_profile, "Frontend developer needed", MockClient())
user_msg2 = captured_prompt['messages'][1]['content']
assert 'Engineer at Facebook' in user_msg2, "Flat past_positions not used as fallback!"
assert 'MIT' in user_msg2, "Flat education not used!"
print("  Flat field fallback works correctly")
print("  PASSED\n")

print("=" * 60)
print("ALL 8 TESTS PASSED!")
print("=" * 60)
