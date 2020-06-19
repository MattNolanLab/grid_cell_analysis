COMMENT
Schmidt-Hieber C, Häusser M (2013) Cellular mechanisms of spatial navigation in the medial entorhinal cortex. Nat Neurosci 16:325-31

17/07/2012
(c) 2012, C. Schmidt-Hieber, University College London
Based on an initial version by Chris Burgess 07/2011

Kinetics based on:
E. Fransen, A. A. Alonso, C. T. Dickson, J. Magistretti, M. E. Hasselmo
Ionic mechanisms in the generation of subthreshold oscillations and 
action potential clustering in entorhinal layer II stellate neurons. 
Hippocampus 14, 368 (2004).

ENDCOMMENT


NEURON {
    SUFFIX ih
    NONSPECIFIC_CURRENT i
    RANGE i, gslow, gfast, gslowbar, gfastbar
    GLOBAL ehcn, taufn, taufdo, taufdd, taufro, taufrd
    GLOBAL tausn, tausdo, tausdd, tausro, tausrd
    GLOBAL mifo, mifd, mife, miso, misd, mise
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
    (mA) = (milliamp)
}

PARAMETER {
    gfastbar = 9.8e-5    (S/cm2)
    gslowbar = 5.3e-5    (S/cm2)
    ehcn    = -20        (mV)
    taufn   = 0.51       (ms)    : original: .51 parameters for tau_fast
    taufdo  = 1.7        (mV)
    taufdd  = 10         (mV)
    taufro    = 340      (mV)
    taufrd    = 52       (mV)
    tausn   = 5.6        (ms)    : parameters for tau_slow
    tausdo  = 17         (mV)
    tausdd  = 14         (mV)
    tausro    = 260      (mV)
    tausrd    = 43       (mV)
    mifo    = 74.2       (mV)    : parameters for steady state m_fast
    mifd    = 9.78       (mV)
    mife    = 1.36
    miso    = 2.83       (mV)    : parameters for steady state m_slow
    misd    = 15.9       (mV)
    mise    = 58.5
}

ASSIGNED {
    v        (mV)
    gslow    (S/cm2)
    gfast    (S/cm2)
    i        (mA/cm2)
    alphaf   (/ms)        : alpha_fast
    betaf    (/ms)        : beta_fast
    alphas   (/ms)        : alpha_slow
    betas    (/ms)        : beta_slow
}

INITIAL {
    : assume steady state
    settables(v)
    mf = alphaf/(alphaf+betaf)
    ms = alphas/(alphas+betas)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gfast = gfastbar*mf
    gslow = gslowbar*ms
    i = (gfast+gslow)*(v-ehcn)
}

STATE {
    mf ms
}

DERIVATIVE states {  
    settables(v)
    mf' = alphaf*(1-mf) - betaf*mf
    ms' = alphas*(1-ms) - betas*ms
}

PROCEDURE settables(v (mV)) { 
    LOCAL mif, mis, tauf, taus 
    TABLE alphaf, betaf, alphas, betas FROM -100 TO 100 WITH 200

    tauf = taufn/( exp( (v-taufdo)/taufdd ) + exp( -(v+taufro)/taufrd ) )
    taus = tausn/( exp( (v-tausdo)/tausdd ) + exp( -(v+tausro)/tausrd ) )
    mif = 1/pow( 1 + exp( (v+mifo)/mifd ), mife )
    mis = 1/pow( 1 + exp( (v+miso)/misd ), mise )

    alphaf = mif/tauf
    alphas = mis/taus
    betaf = (1-mif)/tauf
    betas = (1-mis)/taus
}