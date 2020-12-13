import pandas as pd
import re
import covid_io
from contextlib import suppress


def resolve_geo_class(x):
    return x.country_class if pd.isnull(x.region_class) else f"{x.country_class}_{x.region_class}"


# Initial Data Processing
input_file_name = '01_data.csv'
output_file_name = '02_data.csv'

df = covid_io.read_oxford_data(input_file_name)

# Case Conventions
for name in df.columns:
    df = df.rename(columns={
        name: re.sub(r'(?<!^)(?=[A-Z])', '_', name)
            .lower()
            .replace('/', '_')
            .replace('-', '_')
            .replace(' ', '_')
            .replace('__', '_')
            .replace('__', '_')
            .replace('_code', '_class')
    })

df['geo_class'] = df.apply(resolve_geo_class, axis=1)

# Remove Unused Columns
with suppress(KeyError): df = df.drop(['country_name'], axis=1)
with suppress(KeyError): df = df.drop(['containment_health_index'], axis=1)
with suppress(KeyError): df = df.drop(['economic_support_index'], axis=1)
with suppress(KeyError): df = df.drop(['government_response_index'], axis=1)
with suppress(KeyError): df = df.drop(['m1_wildcard'], axis=1)
with suppress(KeyError): df = df.drop(['region_class'], axis=1)
with suppress(KeyError): df = df.drop(['region_name'], axis=1)
with suppress(KeyError): df = df.drop(['stringency_index'], axis=1)
with suppress(KeyError): df = df.drop(['stringency_legacy_index'], axis=1)
with suppress(KeyError): df = df.drop(['stringency_legacy_index_for_display'], axis=1)

# Add Type Information
df = df.rename(columns={
    "jurisdiction": "jurisdiction_class",
    "date": "date",
    "c1_school_closing": "c1_school_closing_n",
    "c1_flag": "c1_b",
    "c2_workplace_closing": "c2_workplace_closing_n",
    "c2_flag": "c2_b",
    "c3_cancel_public_events": "c3_cancel_public_events_n",
    "c3_flag": "c3_b",
    "c4_restrictions_on_gatherings": "c4_restrictions_on_gatherings_n",
    "c4_flag": "c4_b",
    "c5_close_public_transport": "c5_close_public_transport_n",
    "c5_flag": "c5_b",
    "c6_stay_at_home_requirements": "c6_stay_at_home_requirements_n",
    "c6_flag": "c6_b",
    "c7_restrictions_on_internal_movement": "c7_restrictions_on_internal_movement_n",
    "c7_flag": "c7_b",
    "c8_international_travel_controls": "c8_international_travel_controls_n",
    "e1_income_support": "e1_income_support_n",
    "e1_flag": "e1_b",
    "e2_debt_contract_relief": "e2_debt_contract_relief_n",
    "e3_fiscal_measures": "e3_fiscal_measures_n",
    "e4_international_support": "e4_international_support_n",
    "h1_public_information_campaigns": "h1_public_information_campaigns_n",
    "h1_flag": "h1_b",
    "h2_testing_policy": "h2_testing_policy_n",
    "h3_contact_tracing": "h3_contact_tracing_n",
    "h4_emergency_investment_in_healthcare": "h4_emergency_investment_in_healthcare_n",
    "h5_investment_in_vaccines": "h5_investment_in_vaccines_n",
    "h6_facial_coverings": "h6_facial_coverings_n",
    "h6_flag": "h6_b",
    "h7_vaccination_policy": "h7_vaccination_policy_n",
    "h7_flag": "h7_b",
    "confirmed_cases": "confirmed_cases_n",
    "confirmed_deaths": "confirmed_deaths_n",
    "stringency_index_for_display": "stringency_index_n",
    "government_response_index_for_display": "government_response_index_n",
    "containment_health_index_for_display": "containment_health_index_n",
    "economic_support_index_for_display": "economic_support_index_n"
})

# Sanity Check: Fix any numeric values that may be out of range

for e in df.items():
    name, series = e[0], e[1]
    if series.dtype == 'float64':
        series.clip(lower=0.0, inplace=True)

# Write the dataset
covid_io.write(df, output_file_name)
