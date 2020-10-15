from mediator_width import *
import numpy as np

def determine_gchi_limit_analytical(mediator, mmed, mdm, mu, gq_reference, gchi_reference ):
   '''Convert limit on signal strength mu to a limit on DM coupling gchi.'''

   if(mediator[0].lower()=="a"):
      delta_chi = gamma_axial_chi(mmed,mdm,gchi_reference) / gchi_reference**2
      delta_q = gamma_axial_quark(mmed, gq_reference) / gq_reference**2
      const = gamma_axial_total(mmed,mdm,gq_reference,gchi_reference) / (gq_reference**2 * gchi_reference ** 2)
   elif(mediator[0].lower()=="v"):
      delta_chi = gamma_vector_chi(mmed,mdm,gchi_reference) / gchi_reference**2
      delta_q = gamma_vector_quark(mmed, gq_reference) / gq_reference**2
      const = gamma_vector_total(mmed,mdm,gq_reference,gchi_reference) / (gq_reference**2 * gchi_reference ** 2)
   else:
      raise ValueError("Unknown mediator type specified: '{MED}'. Should be 'A' or 'V'.")

   try:
      g_chi = np.sqrt( delta_q * gq_reference**2 *  mu / ( gq_reference**2 * const - delta_chi * mu ))
   except ValueError:
      return np.inf

   return g_chi

def determine_gq_limit_analytical(mediator,mmed, mdm, mu,gq_reference,gchi_reference,do_plot=False):
   '''Convert limit on signal strength mu to a limit on quark coupling gq.'''
   
   if(mediator[0].lower()=="a"):
      beta = gamma_axial_chi(mmed,mdm) / gchi_reference**2
      alpha = gamma_axial_quark(mmed, gq_reference) / gq_reference**2
      const = gamma_axial_total(mmed,mdm,gq_reference) / (gq_reference**2 * gchi_reference ** 2)
   elif(mediator[0].lower()=="v"):
      beta = gamma_vector_chi(mmed,mdm) / gchi_reference **2
      alpha = gamma_vector_quark(mmed, gq_reference) / gq_reference**2
      const = gamma_vector_total(mmed,mdm,gq_reference) / (gq_reference**2 * gchi_reference ** 2)
   else:
      raise ValueError("Unknown mediator type specified: '{MED}'. Should be 'A' or 'V'.")
   if( beta == 0 ): return None
   gq_sqr = beta * gchi_reference**2 *  mu / ( gchi_reference**2 * const - alpha * mu )
   if(gq_sqr < 0):
      return np.inf
   else:
      return np.sqrt(gq_sqr)

   return gq
