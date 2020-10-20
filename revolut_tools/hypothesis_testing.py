import math
import statsmodels.stats.api as sms
import scipy.stats as st
import pandas as pd

def exp_sample_size_needed(baseline_rate, practical_significance, confidence_level, sensitivity):
    """
    caculate the sample size needed for the parameters of the experiment

    Args:
        baseline_rate: original known conversion level.
        practical_significance: magnitude of the effect desired.
        confidence_level: desired level of statistical confidence in the test.
        sensitivity: statistical power we want on the analysis.

    Returns:
        None
    """
    effect_size = sms.proportion_effectsize(
        baseline_rate, baseline_rate + practical_significance
    )
    sample_size = sms.NormalIndPower().solve_power(
        effect_size = effect_size,
        power = sensitivity,
        alpha = confidence_level,
        ratio=1
    )
    print("Required sample size: ", round(sample_size), " per group")


def test_mean_diff(df, confidence_level=0.05, practical_significance = 0.025):
    """
    Perform a Z test of two populations to assess the null hypothesis of wether
    they come from the same populations.

    Args:
        df: Dataframe with the outcome of the experiment saved on a column 'converted'.
        confidence_level: confidence level of the test that you want to take.
        practical_significance: desired magnitude of the effect

    Returns:
        None
    """
    #Calculate pooled probability
    mask = (df["group"] == "control")
    conversions_control = df["converted"][mask].sum()
    total_users_control = df["converted"][mask].count()

    mask = (df["group"] == "treatment")
    conversions_treatment = df["converted"][mask].sum()
    total_users_treatment = df["converted"][mask].count()

    prob_pooled = (
        (conversions_control + conversions_treatment)
        /
        (total_users_control + total_users_treatment)
        )

    #Calculate pooled standard error and margin of error
    se_pooled = math.sqrt(prob_pooled * (1 - prob_pooled) * (1 / total_users_control + 1 / total_users_treatment))
    z_score = st.norm.ppf(1 - confidence_level / 2)
    margin_of_error = se_pooled * z_score

    #Calculate dhat, the estimated difference between probability of conversions in the experiment and control groups
    d_hat = (conversions_treatment / total_users_treatment) - (conversions_control / total_users_control)

    #Test if we can reject the null hypothesis
    lower_bound = d_hat - margin_of_error
    upper_bound = d_hat + margin_of_error

    if practical_significance < lower_bound:
        print("Reject null hypothesis")
    else:
        print("Do not reject the null hypothesis")

    print("The lower bound of the confidence interval is ", round(lower_bound * 100, 2), "%")
    print("The upper bound of the confidence interval is ", round(upper_bound * 100, 2), "%")
