import pandas as pd
import arviz as az
from sklearn.metrics import root_mean_squared_error, r2_score

def extract_model_predictions(samples, raw_data, scale_min, scale_max, prior = False, hdi = 0.89):
    """
    Extract model predictions given samples from either the prior or posterior.
    """
    # take from prior or posterior
    if prior:
        mean_pred = samples.prior["mu"].mean(dim = ["chain", "draw"]).values
        pred_hdi = az.hdi(samples.prior["mu"], hdi_prob = 0.89)["mu"]
        obs_hdi = az.hdi(samples.prior_predictive["dW"], hdi_prob = 0.89)["dW"]
    else:
        mean_pred = samples.posterior["mu"].mean(dim = ["chain", "draw"]).values
        pred_hdi = az.hdi(samples.posterior["mu"], hdi_prob = 0.89)["mu"]
        obs_hdi = az.hdi(samples.posterior_predictive["dW"], hdi_prob = 0.89)["dW"]

    # collect into a dataframe with the observed data
    mod_pred_df = pd.DataFrame({
        "dW": raw_data["dW"], # observed
        "dW_pred_mean": mean_pred, # mean prediction
        "dW_pred_hdi_lower": pred_hdi.sel(hdi = "lower").values, # lower 89% CI
        "dW_pred_hdi_higher": pred_hdi.sel(hdi = "higher").values, # upper 89% CI
        "dW_obs_hdi_lower": obs_hdi.sel(hdi = "lower").values, # lower 89% CI
        "dW_obs_hdi_higher": obs_hdi.sel(hdi = "higher").values, # upper 89% CI
        "location": raw_data["location"], # location
        "dW_paper": raw_data["dW_pred"] # paper predictions
    })

    # transform back to original scale
    mod_pred_df = rescale_target(mod_pred_df, scale_min, scale_max, target_appends = ["", "_paper", "_pred_mean", "_pred_hdi_lower", "_pred_hdi_higher", "_obs_hdi_lower", "_obs_hdi_higher"])

    return(mod_pred_df)

def rescale_target(data, scale_min_vals, scale_max_vals, target_name = "dW", target_appends = ["", "_pred", "_paper"]):
    """
    Rescale target variable using the scale values.
    """
    data = data.assign(
        scale_min = data["location"].map(scale_min_vals[target_name]),
        scale_max = data["location"].map(scale_max_vals[target_name])
    )
    for col in [target_name + _ for _ in target_appends]:
        data[col] = data[col] * (data["scale_max"] - data["scale_min"]) + data["scale_min"]

    data = data.drop(columns=["scale_min", "scale_max"])
    return data

def calculate_r2_rmse(data, target_name = "dW", target_appends = ["_pred", "_paper"]):
    """
    Calculate R2 score for the target variable.
    """
    r2_scores = {
        "r2" + _ : pd.DataFrame() for _ in target_appends
    }
    rmse_scores = {
        "rmse" + _ : pd.DataFrame() for _ in target_appends
    }

    for this_loc in data["location"].unique():
        this_df = data.loc[data["location"] == this_loc]

        for this_append in target_appends:
            r2_scores["r2" + this_append].loc[this_loc, "r2"] = r2_score(this_df[target_name], this_df[target_name + this_append])
            rmse_scores["rmse" + this_append].loc[this_loc, "rmse"] = root_mean_squared_error(this_df[target_name], this_df[target_name + this_append])
    return r2_scores, rmse_scores