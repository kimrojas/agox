import pytest

strict_tolerances = {
                    'energy':1E-9, 
                    'positions':1E-9
                    }

medium_tolerances = {
                    'energy':1E-6,
                    'positions':1E-6
                    }

lose_tolerances =   {
                    'energy':1E-3,
                    'positions':1E-3,
                    }

tolerance_levels = {
    'lose':lose_tolerances,
    'medium':medium_tolerances,
    'strict':strict_tolerances,
}

def pytest_addoption(parser):
    parser.addoption('--tolerance', type=str, default='lose')

@pytest.fixture
def cmd_options(request):
    return {'tolerance':tolerance_levels[request.config.getoption('tolerance')]}