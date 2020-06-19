COMMENT
km.mod

Mainen ZF, Sejnowski TJ (1996) Influence of dendritic structure on firing pattern in model neocortical neurons. Nature 382:363-6

Potassium channel, Hodgkin-Huxley style kinetics
Based on I-M (muscarinic K channel)
Slow, noninactivating

Author: Zach Mainen, Salk Institute, 1995, zach@salk.edu
	
26 Ago 2002 Modification of original channel to allow 
variable time step and to correct an initialization error.
Done by Michael Hines(michael.hines@yale.e) and 
Ruggero Scorcioni(rscorcio@gmu.edu) at EU Advance Course 
in Computational Neuroscience. Obidos, Portugal

20110202 made threadsafe by Ted Carnevale
20120514 fixed singularity in PROCEDURE rates
ENDCOMMENT

NEURON {
    THREADSAFE
	SUFFIX km
	USEION k READ ek WRITE ik
	RANGE n, gk, gbar
	RANGE ninf, ntau
	GLOBAL Ra, Rb
	GLOBAL q10, temp, tadj, vmin, vmax
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

PARAMETER {
	gbar	   	(mho/cm2)
								
	tha  = -30	(mV)		: v 1/2 for inf
	qa   = 9	(mV)		: inf slope		
	
	Ra   = 0.001	(/ms)		: max act rate  (slow)
	Rb   = 0.001	(/ms)		: max deact rate  (slow)

:	dt		(ms)
	temp = 23	(degC)		: original temp 	
	q10  = 2.3			: temperature sensitivity

	vmin = -120	(mV)
	vmax = 100	(mV)
} 


ASSIGNED {
	v 		(mV)
	celsius		(degC)
	a		(/ms)
	b		(/ms)
	ik 		(mA/cm2)
	gk		(pS/um2)
	ek		(mV)
	ninf
	ntau (ms)	
	tadj
}
 

STATE { n }

INITIAL {
    tadj = q10^((celsius - temp)/(10 (degC))) : make all threads calculate tadj at initialization

	trates(v)
	n = ninf
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	gk = tadj*gbar*n
	ik = (1e-4) * gk * (v - ek)
} 

LOCAL nexp

DERIVATIVE states {   :Computes state variable n 
        trates(v)      :             at the current v and dt.
        n' = (ninf-n)/ntau

}

PROCEDURE trates(v (mV)) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
    TABLE ninf, ntau
    DEPEND celsius, temp, Ra, Rb, tha, qa
    FROM vmin TO vmax WITH 199

	rates(v): not consistently executed from here if usetable_hh == 1

:        tinc = -dt * tadj
:        nexp = 1 - exp(tinc/ntau)
}

UNITSOFF
PROCEDURE rates(v (mV)) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.

    : singular when v = tha
:    a = Ra * (v - tha) / (1 - exp(-(v - tha)/qa))
:    a = Ra * qa*((v - tha)/qa) / (1 - exp(-(v - tha)/qa))
:    a = Ra * qa*(-(v - tha)/qa) / (exp(-(v - tha)/qa) - 1)
    a = Ra * qa * efun(-(v - tha)/qa)

    : singular when v = tha
:    b = -Rb * (v - tha) / (1 - exp((v - tha)/qa))
:    b = -Rb * qa*((v - tha)/qa) / (1 - exp((v - tha)/qa))
:    b = Rb * qa*((v - tha)/qa) / (exp((v - tha)/qa) - 1)
    b = Rb * qa * efun((v - tha)/qa)

        tadj = q10^((celsius - temp)/10)
        ntau = 1/tadj/(a+b)
	ninf = a/(a+b)
}
UNITSON

FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}