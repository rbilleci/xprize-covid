import covid_io
import covid_pandas
import covid_scaler


def scale(dataset, fn_in, fn_out):
    monetary_limit = 1e13
    df = covid_io.read(fn_in, dataset=dataset)
    covid_scaler.scale(df, 'c1_school_closing_n', 0.0, 3.0)
    covid_scaler.scale(df, 'c2_workplace_closing_n', 0.0, 3.0)
    covid_scaler.scale(df, 'c3_cancel_public_events_n', 0.0, 2.0)
    covid_scaler.scale(df, 'c4_restrictions_on_gatherings_n', 0.0, 4.0)
    covid_scaler.scale(df, 'c5_close_public_transport_n', 0.0, 2.0)
    covid_scaler.scale(df, 'c6_stay_at_home_requirements_n', 0.0, 3.0)
    covid_scaler.scale(df, 'c7_restrictions_on_internal_movement_n', 0.0, 2.0)
    covid_scaler.scale(df, 'c8_international_travel_controls_n', 0.0, 4.0)
    covid_scaler.scale(df, 'e1_income_support_n', 0.0, 2.0)
    covid_scaler.scale(df, 'e2_debt_contract_relief_n', 0.0, 2.0)
    covid_scaler.scale(df, 'e3_fiscal_measures_n', 0.0, monetary_limit)
    covid_scaler.scale(df, 'e4_international_support_n', 0.0, monetary_limit)
    covid_scaler.scale(df, 'h1_public_information_campaigns_n', 0.0, 2.0)
    covid_scaler.scale(df, 'h2_testing_policy_n', 0.0, 3.0)
    covid_scaler.scale(df, 'h3_contact_tracing_n', 0.0, 2.0)
    covid_scaler.scale(df, 'h4_emergency_investment_in_healthcare_n', 0.0, monetary_limit)
    covid_scaler.scale(df, 'h5_investment_in_vaccines_n', 0.0, monetary_limit)
    covid_scaler.scale(df, 'h6_facial_coverings_n', 0.0, 4.0)
    covid_scaler.scale(df, 'h7_vaccination_policy_n', 0.0, 5.0)

    # Scale dates
    covid_scaler.scale(df, 'date_days_since_start_of_calendar_n_nn', 0.0, 4000 * 365)  # -> will center around 0.5ish
    covid_scaler.scale(df, 'date_days_since_first_known_infection_n_nn', 0,
                       10 * 365)  # -> support prediction up to 10 years

    # Scale the indexes
    covid_scaler.scale(df, 'stringency_index_n', 0.0, 100.0)
    covid_scaler.scale(df, 'government_response_index_n', 0.0, 100.0)
    covid_scaler.scale(df, 'containment_health_index_n', 0.0, 100.0)
    covid_scaler.scale(df, 'economic_support_index_n', 0.0, 100.0)

    # Scale all sin / cos fields
    for e in df.items():
        name, series = e[0], e[1]
        if name.endswith('_sin') or name.endswith('_cos'):
            covid_scaler.scale(df, name, 0.0, 1.0)

    covid_io.write(df, fn_out, dataset=dataset)


covid_pandas.configure()
input_file_name = '11_data.csv'
output_file_name = '12_data.csv'

scale('train', input_file_name, output_file_name)
scale('validation', input_file_name, output_file_name)
scale('test', input_file_name, output_file_name)
