import numpy as np
from scipy.optimize import curve_fit
import sympy as sp
from scipy.optimize import least_squares

# oxygen - Fit

def stern_volmer_two_site_tau(Q, tau0, Ksv1, Ksv2, f):
    return tau0 * (
        f / (1 + Ksv1 * Q)
        + (1 - f) / (1 + Ksv2 * Q)
    )

def stern_volmer_partial_tau(Q, tau0, Ksv1, f):
    return tau0 * (
        f / (1 + Ksv1 * Q) + (1 - f)
    )

def fit_stern_volmer(Q, tau, force_ksv2_zero=False, fixed_f=None):

    Q = np.asarray(Q, dtype=float)
    tau = np.asarray(tau, dtype=float)

    if Q.size < 3:
        raise ValueError("At least 3 calibration points required")

    def model(Q, tau0, Ksv1, Ksv2, f):
        return tau0 * (
            f / (1 + Ksv1 * Q)
            + (1 - f) / (1 + Ksv2 * Q)
        )

    f_fixed = fixed_f is not None

    if force_ksv2_zero:

        if f_fixed:
            print("partial (f fixed)")

            def model(Q, tau0, Ksv1):
                return stern_volmer_partial_tau(Q, tau0, Ksv1, fixed_f)

            popt, pcov = curve_fit(
                model,
                Q,
                tau,
                p0=[np.max(tau), 0.05],
                bounds=([0, 0], [np.inf, np.inf])
            )

            tau0, Ksv1 = popt
            Ksv2 = 0.0
            f = fixed_f

        else:
            print("partial (full fit)")

            popt, pcov = curve_fit(
                stern_volmer_partial_tau,
                Q,
                tau,
                p0=[np.max(tau), 0.05, 0.5],
                bounds=([0, 0, 0], [np.inf, np.inf, 1])
            )

            tau0, Ksv1, f = popt
            Ksv2 = 0.0

        model_func = stern_volmer_partial_tau


    else:

        if f_fixed:
            print("two-site (f fixed)")

            def model(Q, tau0, Ksv1, Ksv2):
                return stern_volmer_two_site_tau(Q, tau0, Ksv1, Ksv2, fixed_f)

            popt, pcov = curve_fit(
                model,
                Q,
                tau,
                p0=[np.max(tau), 0.05, 0.005],
                bounds=([0, 0, 1e-6], [np.inf, np.inf, np.inf])
            )

            tau0, Ksv1, Ksv2 = popt
            f = fixed_f

        else:
            print("two-site (full fit)")

            popt, pcov = curve_fit(
                stern_volmer_two_site_tau,
                Q,
                tau,
                p0=[np.max(tau), 0.2, 0.001, 0.7],
                bounds=([0, 0, 1e-6, 0], [np.inf, np.inf, np.inf, 1])
            )

            tau0, Ksv1, Ksv2, f = popt

        model_func = stern_volmer_two_site_tau

    try:
        perr = np.sqrt(np.diag(pcov))
    except:
        perr = []

    return {
        "Ksv1": Ksv1,
        "Ksv2": Ksv2,
        "f": f,
        "tau0": tau0,
        "errors": perr,
        "model": model_func,
        "partial": force_ksv2_zero
    }

tau_s, pO2_s, tau0_s, Ksv1_s, Ksv2_s, f_s = sp.symbols(
    "tau pO2 tau0 Ksv1 Ksv2 f", positive=True
)

expr = (
    tau0_s * (
        f_s / (1 + Ksv1_s * pO2_s)
        + (1 - f_s) / (1 + Ksv2_s * pO2_s)
    )
    - tau_s
)

solutions = sp.solve(expr, pO2_s) 
physical_root = solutions[1]

_pO2_relation = sp.lambdify(
    (tau_s, tau0_s, Ksv1_s, Ksv2_s, f_s),
    physical_root,
    modules="numpy"
)


def pO2_from_tau(tau, tau0, Ksv1, Ksv2, f):
    tau = np.asarray(tau, dtype=float)
    return _pO2_relation(tau, tau0, Ksv1, Ksv2, f)

def pO2_from_tau_partial(tau, tau0, Ksv1, f):
    y = tau / tau0
    return (f / (y - (1 - f)) - 1) / Ksv1

# temperature - Fit

kB = 8.617333262e-5

def arrhenius_tau(T, k0, k1, deltaE):

    return 1.0 / (k0 + k1 * np.exp(-deltaE / (kB * T)))

def fit_arrhenius(T, tau):

    T = np.asarray(T, dtype=float)
    tau = np.asarray(tau, dtype=float)

    if T.size < 3:
        raise ValueError("At least 3 calibration points required")

    k0_0 = 1 / np.max(tau)
    k1_0 = k0_0
    deltaE_0 = 0.1  

    popt, pcov = curve_fit(
        arrhenius_tau,
        T,
        tau,
        p0=[k0_0, k1_0, deltaE_0],
        bounds=([0, 0, 0], [np.inf, np.inf, 1.0])
    )

    k0, k1, deltaE = popt
    perr = np.sqrt(np.diag(pcov))

    return {
        "k0": k0,
        "k1": k1,
        "deltaE": deltaE,
        "errors": perr,
        "model": arrhenius_tau
    }

def T_from_tau_arrhenius(tau, k0, k1, deltaE):

    tau = np.asarray(tau, dtype=float)

    rate = 1.0 / tau - k0
    rate = np.clip(rate, 1e-12, None)

    T = -deltaE / (kB * np.log(rate / k1))
    return T

# pH - Fit

def boltzmann_pH(pH, Top, Bottom, pKa, slope):

    return Bottom + (Top - Bottom) / (1.0 + 10.0 ** ((pH - pKa) / slope))

def fit_boltzmann_pH(pH, ratio):
    pH = np.asarray(pH, dtype=float)
    ratio = np.asarray(ratio, dtype=float)

    if pH.size < 4:
        raise ValueError("At least 4 calibration points required")

    r_min = np.min(ratio)
    r_max = np.max(ratio)

    Bottom0 = r_min
    Top0    = r_max

    corr = np.corrcoef(pH, ratio)[0, 1]
    slope_sign = -1.0 if corr > 0 else 1.0

    pKa0   = pH[np.argmin(np.abs(ratio - (Top0 + Bottom0) / 2))]
    slope0 = slope_sign * 1.0

    popt, pcov = curve_fit(
        boltzmann_pH,
        pH,
        ratio,
        p0=[Top0, Bottom0, pKa0, slope0],
        bounds=(
            [0, 0, 0, -5.0],    
            [np.inf, np.inf, 14, 5.0]
        )
    )

    Top, Bottom, pKa, slope = popt

    if Top < Bottom:
        Top, Bottom = Bottom, Top
        slope = -slope

    perr = np.sqrt(np.diag(pcov))

    return {
        "Top": Top,
        "Bottom": Bottom,
        "pKa": pKa,
        "slope": slope,
        "errors": perr,
        "model": boltzmann_pH
    }


def pH_from_ratio_boltz(R, Top, Bottom, pKa, slope):
    R = np.asarray(R, dtype=float)

    eps = 1e-9
    R = np.clip(R, Bottom + eps, Top - eps)

    return pKa + slope * np.log10((Top - R) / (R - Bottom))

# oxygen - 2 point recalibration

def recalibrate_tau0_ksv1(
    tau_ref,
    Q_ref,
    Ksv2,
    f,
    tau0_init,
    Ksv1_init
):

    tau_ref = np.asarray(tau_ref, dtype=float)
    Q_ref = np.asarray(Q_ref, dtype=float)

    if tau_ref.size != 2 or Q_ref.size != 2:
        raise ValueError("Exactly 2 calibration points required")

    def model(Q, tau0, Ksv1):
        return tau0 * (
            f / (1 + Ksv1 * Q)
            + (1 - f) / (1 + Ksv2 * Q)
        )

    def residuals(params):
        tau0, Ksv1 = params
        return model(Q_ref, tau0, Ksv1) - tau_ref

    result = least_squares(
        residuals,
        x0=[tau0_init, Ksv1_init],
        bounds=([0, 0], [np.inf, np.inf])
    )

    tau0_new, Ksv1_new = result.x

    return {
        "tau0": tau0_new,
        "Ksv1": Ksv1_new,
        "success": result.success,
        "cost": result.cost
    }

# pH - 3 point recalibration

def recalibrate_pH(
    pH_ref,
    R_ref,
    Top_init,
    Bottom_init,
    pKa_init,
    slope,
    weight_prior=0.0
):
    """
    3+ points:
        Bottom + Top fixed from edge points
        fit pKa only

    2 points:
        fit Top, Bottom, pKa
    """

    pH_ref = np.asarray(pH_ref, dtype=float)
    R_ref  = np.asarray(R_ref, dtype=float)

    if len(pH_ref) < 2:
        raise ValueError("Need at least 2 recalibration points")

    idx = np.argsort(pH_ref)
    pH_ref = pH_ref[idx]
    R_ref  = R_ref[idx]

    # 3 points

    if len(pH_ref) >= 3:

        Bottom = R_ref[0]  
        Top    = R_ref[-1] 

        pH_mid = pH_ref[1:-1]
        R_mid  = R_ref[1:-1]

        def model(pH, pKa):
            return Bottom + (Top - Bottom) / (
                1.0 + 10.0 ** ((pH - pKa) / slope)
            )

        def residuals(x):
            pKa = x[0]
            return model(pH_mid, pKa) - R_mid

        result = least_squares(
            residuals,
            x0=[pKa_init]
        )

        pKa = result.x[0]

    # 2 points 

    else:

        def model(pH, Top, Bottom, pKa):
            return Bottom + (Top - Bottom) / (
                1.0 + 10.0 ** ((pH - pKa) / slope)
            )

        def residuals(x):
            Top, Bottom, pKa = x

            r_fit = model(pH_ref, Top, Bottom, pKa) - R_ref

            r_prior = weight_prior * np.array([
                Top - Top_init,
                Bottom - Bottom_init,
                pKa - pKa_init
            ])

            return np.concatenate([r_fit, r_prior])

        result = least_squares(
            residuals,
            x0=[Top_init, Bottom_init, pKa_init]
        )

        Top, Bottom, pKa = result.x

    return {
        "Top": Top,
        "Bottom": Bottom,
        "pKa": pKa,
        "success": True
    }

# temperature - 2 point recalibration

def recalibrate_arrhenius_k0_k1(
    tau_ref,
    T_ref,
    deltaE,
    k0_init,
    k1_init
):
    """
    2-point recalibration for Arrhenius model:
    solves for k0 and k1, keeps deltaE fixed
    """

    tau_ref = np.asarray(tau_ref, dtype=float)
    T_ref = np.asarray(T_ref, dtype=float)

    if tau_ref.size != 2 or T_ref.size != 2:
        raise ValueError("Need exactly 2 reference points")

    def residuals(params):
        k0, k1 = params

        tau_model = 1.0 / (
            k0 + k1 * np.exp(-deltaE / (kB * T_ref))
        )

        return tau_model - tau_ref

    result = least_squares(
        residuals,
        x0=[k0_init, k1_init],
        bounds=(0, np.inf)
    )

    k0, k1 = result.x

    return {
        "k0": k0,
        "k1": k1,
        "deltaE": deltaE
    }