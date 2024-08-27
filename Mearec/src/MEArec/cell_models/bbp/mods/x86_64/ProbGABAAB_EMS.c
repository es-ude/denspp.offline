/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__ProbGABAAB_EMS
#define _nrn_initial _nrn_initial__ProbGABAAB_EMS
#define nrn_cur _nrn_cur__ProbGABAAB_EMS
#define _nrn_current _nrn_current__ProbGABAAB_EMS
#define nrn_jacob _nrn_jacob__ProbGABAAB_EMS
#define nrn_state _nrn_state__ProbGABAAB_EMS
#define _net_receive _net_receive__ProbGABAAB_EMS 
#define setRNG setRNG__ProbGABAAB_EMS 
#define state state__ProbGABAAB_EMS 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define tau_r_GABAA _p[0]
#define tau_r_GABAA_columnindex 0
#define tau_d_GABAA _p[1]
#define tau_d_GABAA_columnindex 1
#define tau_r_GABAB _p[2]
#define tau_r_GABAB_columnindex 2
#define tau_d_GABAB _p[3]
#define tau_d_GABAB_columnindex 3
#define Use _p[4]
#define Use_columnindex 4
#define Dep _p[5]
#define Dep_columnindex 5
#define Fac _p[6]
#define Fac_columnindex 6
#define e_GABAA _p[7]
#define e_GABAA_columnindex 7
#define e_GABAB _p[8]
#define e_GABAB_columnindex 8
#define u0 _p[9]
#define u0_columnindex 9
#define synapseID _p[10]
#define synapseID_columnindex 10
#define verboseLevel _p[11]
#define verboseLevel_columnindex 11
#define GABAB_ratio _p[12]
#define GABAB_ratio_columnindex 12
#define i _p[13]
#define i_columnindex 13
#define i_GABAA _p[14]
#define i_GABAA_columnindex 14
#define i_GABAB _p[15]
#define i_GABAB_columnindex 15
#define g_GABAA _p[16]
#define g_GABAA_columnindex 16
#define g_GABAB _p[17]
#define g_GABAB_columnindex 17
#define A_GABAA_step _p[18]
#define A_GABAA_step_columnindex 18
#define B_GABAA_step _p[19]
#define B_GABAA_step_columnindex 19
#define A_GABAB_step _p[20]
#define A_GABAB_step_columnindex 20
#define B_GABAB_step _p[21]
#define B_GABAB_step_columnindex 21
#define g _p[22]
#define g_columnindex 22
#define Rstate _p[23]
#define Rstate_columnindex 23
#define tsyn_fac _p[24]
#define tsyn_fac_columnindex 24
#define u _p[25]
#define u_columnindex 25
#define A_GABAA _p[26]
#define A_GABAA_columnindex 26
#define B_GABAA _p[27]
#define B_GABAA_columnindex 27
#define A_GABAB _p[28]
#define A_GABAB_columnindex 28
#define B_GABAB _p[29]
#define B_GABAB_columnindex 29
#define factor_GABAA _p[30]
#define factor_GABAA_columnindex 30
#define factor_GABAB _p[31]
#define factor_GABAB_columnindex 31
#define DA_GABAA _p[32]
#define DA_GABAA_columnindex 32
#define DB_GABAA _p[33]
#define DB_GABAA_columnindex 33
#define DA_GABAB _p[34]
#define DA_GABAB_columnindex 34
#define DB_GABAB _p[35]
#define DB_GABAB_columnindex 35
#define v _p[36]
#define v_columnindex 36
#define _g _p[37]
#define _g_columnindex 37
#define _tsav _p[38]
#define _tsav_columnindex 38
#define _nd_area  *_ppvar[0]._pval
#define rng	*_ppvar[2]._pval
#define _p_rng	_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  2;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_setRNG(void*);
 static double _hoc_state(void*);
 static double _hoc_toggleVerbose(void*);
 static double _hoc_urand(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "setRNG", _hoc_setRNG,
 "state", _hoc_state,
 "toggleVerbose", _hoc_toggleVerbose,
 "urand", _hoc_urand,
 0, 0
};
#define toggleVerbose toggleVerbose_ProbGABAAB_EMS
#define urand urand_ProbGABAAB_EMS
 extern double toggleVerbose( _threadargsproto_ );
 extern double urand( _threadargsproto_ );
 /* declare global and static user variables */
#define gmax gmax_ProbGABAAB_EMS
 double gmax = 0.001;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gmax_ProbGABAAB_EMS", "uS",
 "tau_r_GABAA", "ms",
 "tau_d_GABAA", "ms",
 "tau_r_GABAB", "ms",
 "tau_d_GABAB", "ms",
 "Use", "1",
 "Dep", "ms",
 "Fac", "ms",
 "e_GABAA", "mV",
 "e_GABAB", "mV",
 "GABAB_ratio", "1",
 "i", "nA",
 "i_GABAA", "nA",
 "i_GABAB", "nA",
 "g_GABAA", "uS",
 "g_GABAB", "uS",
 "g", "uS",
 "Rstate", "1",
 "tsyn_fac", "ms",
 "u", "1",
 0,0
};
 static double A_GABAB0 = 0;
 static double A_GABAA0 = 0;
 static double B_GABAB0 = 0;
 static double B_GABAA0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "gmax_ProbGABAAB_EMS", &gmax_ProbGABAAB_EMS,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ProbGABAAB_EMS",
 "tau_r_GABAA",
 "tau_d_GABAA",
 "tau_r_GABAB",
 "tau_d_GABAB",
 "Use",
 "Dep",
 "Fac",
 "e_GABAA",
 "e_GABAB",
 "u0",
 "synapseID",
 "verboseLevel",
 "GABAB_ratio",
 0,
 "i",
 "i_GABAA",
 "i_GABAB",
 "g_GABAA",
 "g_GABAB",
 "A_GABAA_step",
 "B_GABAA_step",
 "A_GABAB_step",
 "B_GABAB_step",
 "g",
 "Rstate",
 "tsyn_fac",
 "u",
 0,
 "A_GABAA",
 "B_GABAA",
 "A_GABAB",
 "B_GABAB",
 0,
 "rng",
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 39, _prop);
 	/*initialize range parameters*/
 	tau_r_GABAA = 0.2;
 	tau_d_GABAA = 8;
 	tau_r_GABAB = 3.5;
 	tau_d_GABAB = 260.9;
 	Use = 1;
 	Dep = 100;
 	Fac = 10;
 	e_GABAA = -80;
 	e_GABAB = -97;
 	u0 = 0;
 	synapseID = 0;
 	verboseLevel = 0;
 	GABAB_ratio = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 39;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 static void _net_receive(Point_process*, double*, double);
 static void _net_init(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ProbGABAAB_EMS_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 39, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "pointer");
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 5;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ProbGABAAB_EMS /home/mahmoud/MEArec/src/MEArec/cell_models/bbp/mods/ProbGABAAB_EMS.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "GABAAB receptor with presynaptic short-term plasticity ";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int setRNG(_threadargsproto_);
static int state(_threadargsproto_);
 
/*VERBATIM*/
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);

 
static int  state ( _threadargsproto_ ) {
   A_GABAA = A_GABAA * A_GABAA_step ;
   B_GABAA = B_GABAA * B_GABAA_step ;
   A_GABAB = A_GABAB * A_GABAB_step ;
   B_GABAB = B_GABAB * B_GABAB_step ;
    return 0; }
 
static double _hoc_state(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 state ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
   double _lresult ;
 _args[1] = _args[0] ;
   _args[2] = _args[0] * GABAB_ratio ;
   if (  ! ( _args[0] > 0.0 ) ) {
     
/*VERBATIM*/
        return;
 }
   if ( Fac > 0.0 ) {
     u = u * exp ( - ( t - tsyn_fac ) / Fac ) ;
     }
   else {
     u = Use ;
     }
   if ( Fac > 0.0 ) {
     u = u + Use * ( 1.0 - u ) ;
     }
   tsyn_fac = t ;
   if ( Rstate  == 0.0 ) {
     _args[3] = exp ( - ( t - _args[4] ) / Dep ) ;
     _lresult = urand ( _threadargs_ ) ;
     if ( _lresult > _args[3] ) {
       Rstate = 1.0 ;
       if ( verboseLevel > 0.0 ) {
         printf ( "Recovered! %f at time %g: Psurv = %g, urand=%g\n" , synapseID , t , _args[3] , _lresult ) ;
         }
       }
     else {
       _args[4] = t ;
       if ( verboseLevel > 0.0 ) {
         printf ( "Failed to recover! %f at time %g: Psurv = %g, urand=%g\n" , synapseID , t , _args[3] , _lresult ) ;
         }
       }
     }
   if ( Rstate  == 1.0 ) {
     _lresult = urand ( _threadargs_ ) ;
     if ( _lresult < u ) {
       _args[4] = t ;
       Rstate = 0.0 ;
       A_GABAA = A_GABAA + _args[1] * factor_GABAA ;
       B_GABAA = B_GABAA + _args[1] * factor_GABAA ;
       A_GABAB = A_GABAB + _args[2] * factor_GABAB ;
       B_GABAB = B_GABAB + _args[2] * factor_GABAB ;
       if ( verboseLevel > 0.0 ) {
         printf ( "Release! %f at time %g: vals %g %g %g \n" , synapseID , t , A_GABAA , _args[1] , factor_GABAA ) ;
         }
       }
     else {
       if ( verboseLevel > 0.0 ) {
         printf ( "Failure! %f at time %g: urand = %g\n" , synapseID , t , _lresult ) ;
         }
       }
     }
   } }
 
static void _net_init(Point_process* _pnt, double* _args, double _lflag) {
       double* _p = _pnt->_prop->param;
    Datum* _ppvar = _pnt->_prop->dparam;
    Datum* _thread = (Datum*)0;
    NrnThread* _nt = (NrnThread*)_pnt->_vnt;
 _args[4] = t ;
   }
 
static int  setRNG ( _threadargsproto_ ) {
   
/*VERBATIM*/
    {
        /**
         * This function takes a NEURON Random object declared in hoc and makes it usable by this mod file.
         * Note that this method is taken from Brett paper as used by netstim.hoc and netstim.mod
         */
        void** pv = (void**)(&_p_rng);
        if( ifarg(1)) {
            *pv = nrn_random_arg(1);
        } else {
            *pv = (void*)0;
        }
    }
  return 0; }
 
static double _hoc_setRNG(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 setRNG ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
double urand ( _threadargsproto_ ) {
   double _lurand;
 
/*VERBATIM*/
        double value;
        if (_p_rng) {
                /*
                :Supports separate independent but reproducible streams for
                : each instance. However, the corresponding hoc Random
                : distribution MUST be set to Random.uniform(1)
                */
                value = nrn_random_pick(_p_rng);
                //printf("random stream for this simulation = %lf\n",value);
                return value;
        }else{
 _lurand = scop_random ( 1.0 ) ;
   
/*VERBATIM*/
        }
 _lurand = value ;
   
return _lurand;
 }
 
static double _hoc_urand(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  urand ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
double toggleVerbose ( _threadargsproto_ ) {
   double _ltoggleVerbose;
 verboseLevel = 1.0 - verboseLevel ;
   
return _ltoggleVerbose;
 }
 
static double _hoc_toggleVerbose(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  toggleVerbose ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int _ode_count(int _type){ hoc_execerror("ProbGABAAB_EMS", "cannot be used with CVODE"); return 0;}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  A_GABAB = A_GABAB0;
  A_GABAA = A_GABAA0;
  B_GABAB = B_GABAB0;
  B_GABAA = B_GABAA0;
 {
   double _ltp_GABAA , _ltp_GABAB ;
 Rstate = 1.0 ;
   tsyn_fac = 0.0 ;
   u = u0 ;
   A_GABAA = 0.0 ;
   B_GABAA = 0.0 ;
   A_GABAB = 0.0 ;
   B_GABAB = 0.0 ;
   _ltp_GABAA = ( tau_r_GABAA * tau_d_GABAA ) / ( tau_d_GABAA - tau_r_GABAA ) * log ( tau_d_GABAA / tau_r_GABAA ) ;
   _ltp_GABAB = ( tau_r_GABAB * tau_d_GABAB ) / ( tau_d_GABAB - tau_r_GABAB ) * log ( tau_d_GABAB / tau_r_GABAB ) ;
   factor_GABAA = - exp ( - _ltp_GABAA / tau_r_GABAA ) + exp ( - _ltp_GABAA / tau_d_GABAA ) ;
   factor_GABAA = 1.0 / factor_GABAA ;
   factor_GABAB = - exp ( - _ltp_GABAB / tau_r_GABAB ) + exp ( - _ltp_GABAB / tau_d_GABAB ) ;
   factor_GABAB = 1.0 / factor_GABAB ;
   A_GABAA_step = exp ( dt * ( ( - 1.0 ) / tau_r_GABAA ) ) ;
   B_GABAA_step = exp ( dt * ( ( - 1.0 ) / tau_d_GABAA ) ) ;
   A_GABAB_step = exp ( dt * ( ( - 1.0 ) / tau_r_GABAB ) ) ;
   B_GABAB_step = exp ( dt * ( ( - 1.0 ) / tau_d_GABAB ) ) ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   g_GABAA = gmax * ( B_GABAA - A_GABAA ) ;
   g_GABAB = gmax * ( B_GABAB - A_GABAB ) ;
   g = g_GABAA + g_GABAB ;
   i_GABAA = g_GABAA * ( v - e_GABAA ) ;
   i_GABAB = g_GABAB * ( v - e_GABAB ) ;
   i = i_GABAA + i_GABAB ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 {  { state(_p, _ppvar, _thread, _nt); }
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/mahmoud/MEArec/src/MEArec/cell_models/bbp/mods/ProbGABAAB_EMS.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "/*                                                                               \n"
  "Copyright (c) 2015 EPFL-BBP, All rights reserved.                                \n"
  "                                                                                 \n"
  "THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''                    \n"
  "AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,            \n"
  "THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR           \n"
  "PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT                 \n"
  "BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR           \n"
  "CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF             \n"
  "SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                  \n"
  "BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,            \n"
  "WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE             \n"
  "OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN           \n"
  "IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    \n"
  "                                                                                 \n"
  "This work is licensed under a                                                    \n"
  "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. \n"
  "To view a copy of this license, visit                                            \n"
  "http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode or send a letter to   \n"
  "Creative Commons,                                                                \n"
  "171 Second Street, Suite 300,                                                    \n"
  "San Francisco, California, 94105, USA.                                           \n"
  "*/                                                                               \n"
  "ENDCOMMENT\n"
  "\n"
  "TITLE GABAAB receptor with presynaptic short-term plasticity \n"
  "\n"
  "\n"
  "COMMENT\n"
  "GABAA receptor conductance using a dual-exponential profile\n"
  "presynaptic short-term plasticity based on Fuhrmann et al, 2002\n"
  "Implemented by Srikanth Ramaswamy, Blue Brain Project, March 2009\n"
  "\n"
  "_EMS (Eilif Michael Srikanth)\n"
  "Modification of ProbGABAA: 2-State model by Eilif Muller, Michael Reimann, Srikanth Ramaswamy, Blue Brain Project, August 2011\n"
  "This new model was motivated by the following constraints:\n"
  "\n"
  "1) No consumption on failure.  \n"
  "2) No release just after release until recovery.\n"
  "3) Same ensemble averaged trace as deterministic/canonical Tsodyks-Markram \n"
  "   using same parameters determined from experiment.\n"
  "4) Same quantal size as present production probabilistic model.\n"
  "\n"
  "To satisfy these constaints, the synapse is implemented as a\n"
  "uni-vesicular (generalization to multi-vesicular should be\n"
  "straight-forward) 2-state Markov process.  The states are\n"
  "{1=recovered, 0=unrecovered}.\n"
  "\n"
  "For a pre-synaptic spike or external spontaneous release trigger\n"
  "event, the synapse will only release if it is in the recovered state,\n"
  "and with probability u (which follows facilitation dynamics).  If it\n"
  "releases, it will transition to the unrecovered state.  Recovery is as\n"
  "a Poisson process with rate 1/Dep.\n"
  "\n"
  "This model satisfies all of (1)-(4).\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "COMMENT\n"
  "/**\n"
  " @file ProbGABAAB_EMS.mod\n"
  " @brief GABAAB receptor with presynaptic short-term plasticity\n"
  " @author Eilif Muller, Michael Reimann, Srikanth Ramaswamy, James King @ BBP\n"
  " @date 2011\n"
  "*/\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "    THREADSAFE\n"
  "	POINT_PROCESS ProbGABAAB_EMS\n"
  "	RANGE tau_r_GABAA, tau_d_GABAA, tau_r_GABAB, tau_d_GABAB \n"
  "	RANGE Use, u, Dep, Fac, u0, Rstate, tsyn_fac, u\n"
  "	RANGE i,i_GABAA, i_GABAB, g_GABAA, g_GABAB, g, e_GABAA, e_GABAB, GABAB_ratio\n"
  "        RANGE A_GABAA_step, B_GABAA_step, A_GABAB_step, B_GABAB_step\n"
  "	NONSPECIFIC_CURRENT i\n"
  "    POINTER rng\n"
  "    RANGE synapseID, verboseLevel\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	tau_r_GABAA  = 0.2   (ms)  : dual-exponential conductance profile\n"
  "	tau_d_GABAA = 8   (ms)  : IMPORTANT: tau_r < tau_d\n"
  "    tau_r_GABAB  = 3.5   (ms)  : dual-exponential conductance profile :Placeholder value from hippocampal recordings SR\n"
  "	tau_d_GABAB = 260.9   (ms)  : IMPORTANT: tau_r < tau_d  :Placeholder value from hippocampal recordings \n"
  "	Use        = 1.0   (1)   : Utilization of synaptic efficacy (just initial values! Use, Dep and Fac are overwritten by BlueBuilder assigned values) \n"
  "	Dep   = 100   (ms)  : relaxation time constant from depression\n"
  "	Fac   = 10   (ms)  :  relaxation time constant from facilitation\n"
  "	e_GABAA    = -80     (mV)  : GABAA reversal potential\n"
  "    e_GABAB    = -97     (mV)  : GABAB reversal potential\n"
  "    gmax = .001 (uS) : weight conversion factor (from nS to uS)\n"
  "    u0 = 0 :initial value of u, which is the running value of release probability\n"
  "    synapseID = 0\n"
  "    verboseLevel = 0\n"
  "	GABAB_ratio = 0 (1) : The ratio of GABAB to GABAA\n"
  "}\n"
  "\n"
  "COMMENT\n"
  "The Verbatim block is needed to generate random nos. from a uniform distribution between 0 and 1 \n"
  "for comparison with Pr to decide whether to activate the synapse or not\n"
  "ENDCOMMENT\n"
  "   \n"
  "VERBATIM\n"
  "#include<stdlib.h>\n"
  "#include<stdio.h>\n"
  "#include<math.h>\n"
  "\n"
  "double nrn_random_pick(void* r);\n"
  "void* nrn_random_arg(int argpos);\n"
  "\n"
  "ENDVERBATIM\n"
  "  \n"
  "\n"
  "ASSIGNED {\n"
  "	v (mV)\n"
  "	i (nA)\n"
  "        i_GABAA (nA)\n"
  "        i_GABAB (nA)\n"
  "        g_GABAA (uS)\n"
  "        g_GABAB (uS)\n"
  "        A_GABAA_step\n"
  "        B_GABAA_step\n"
  "        A_GABAB_step\n"
  "        B_GABAB_step\n"
  "	g (uS)\n"
  "	factor_GABAA\n"
  "        factor_GABAB\n"
  "        rng\n"
  "\n"
  "       : Recording these three, you can observe full state of model\n"
  "       : tsyn_fac gives you presynaptic times, Rstate gives you \n"
  "	 : state transitions,\n"
  "	 : u gives you the \"release probability\" at transitions \n"
  "	 : (attention: u is event based based, so only valid at incoming events)\n"
  "       Rstate (1) : recovered state {0=unrecovered, 1=recovered}\n"
  "       tsyn_fac (ms) : the time of the last spike\n"
  "       u (1) : running release probability\n"
  "\n"
  "\n"
  "}\n"
  "\n"
  "STATE {\n"
  "        A_GABAA       : GABAA state variable to construct the dual-exponential profile - decays with conductance tau_r_GABAA\n"
  "        B_GABAA       : GABAA state variable to construct the dual-exponential profile - decays with conductance tau_d_GABAA\n"
  "        A_GABAB       : GABAB state variable to construct the dual-exponential profile - decays with conductance tau_r_GABAB\n"
  "        B_GABAB       : GABAB state variable to construct the dual-exponential profile - decays with conductance tau_d_GABAB\n"
  "}\n"
  "\n"
  "INITIAL{\n"
  "\n"
  "        LOCAL tp_GABAA, tp_GABAB\n"
  "\n"
  "	Rstate=1\n"
  "	tsyn_fac=0\n"
  "	u=u0\n"
  "        \n"
  "        A_GABAA = 0\n"
  "        B_GABAA = 0\n"
  "        \n"
  "        A_GABAB = 0\n"
  "        B_GABAB = 0\n"
  "        \n"
  "        tp_GABAA = (tau_r_GABAA*tau_d_GABAA)/(tau_d_GABAA-tau_r_GABAA)*log(tau_d_GABAA/tau_r_GABAA) :time to peak of the conductance\n"
  "        tp_GABAB = (tau_r_GABAB*tau_d_GABAB)/(tau_d_GABAB-tau_r_GABAB)*log(tau_d_GABAB/tau_r_GABAB) :time to peak of the conductance\n"
  "        \n"
  "        factor_GABAA = -exp(-tp_GABAA/tau_r_GABAA)+exp(-tp_GABAA/tau_d_GABAA) :GABAA Normalization factor - so that when t = tp_GABAA, gsyn = gpeak\n"
  "        factor_GABAA = 1/factor_GABAA\n"
  "        \n"
  "        factor_GABAB = -exp(-tp_GABAB/tau_r_GABAB)+exp(-tp_GABAB/tau_d_GABAB) :GABAB Normalization factor - so that when t = tp_GABAB, gsyn = gpeak\n"
  "        factor_GABAB = 1/factor_GABAB\n"
  "        \n"
  "        A_GABAA_step = exp(dt*(( - 1.0 ) / tau_r_GABAA))\n"
  "        B_GABAA_step = exp(dt*(( - 1.0 ) / tau_d_GABAA))\n"
  "        A_GABAB_step = exp(dt*(( - 1.0 ) / tau_r_GABAB))\n"
  "        B_GABAB_step = exp(dt*(( - 1.0 ) / tau_d_GABAB))\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE state\n"
  "	\n"
  "        g_GABAA = gmax*(B_GABAA-A_GABAA) :compute time varying conductance as the difference of state variables B_GABAA and A_GABAA\n"
  "        g_GABAB = gmax*(B_GABAB-A_GABAB) :compute time varying conductance as the difference of state variables B_GABAB and A_GABAB \n"
  "        g = g_GABAA + g_GABAB\n"
  "        i_GABAA = g_GABAA*(v-e_GABAA) :compute the GABAA driving force based on the time varying conductance, membrane potential, and GABAA reversal\n"
  "        i_GABAB = g_GABAB*(v-e_GABAB) :compute the GABAB driving force based on the time varying conductance, membrane potential, and GABAB reversal\n"
  "        i = i_GABAA + i_GABAB\n"
  "}\n"
  "\n"
  "PROCEDURE state() {\n"
  "        A_GABAA = A_GABAA*A_GABAA_step\n"
  "        B_GABAA = B_GABAA*B_GABAA_step\n"
  "        A_GABAB = A_GABAB*A_GABAB_step\n"
  "        B_GABAB = B_GABAB*B_GABAB_step\n"
  "}\n"
  "\n"
  "\n"
  "NET_RECEIVE (weight, weight_GABAA, weight_GABAB, Psurv, tsyn (ms)){\n"
  "    LOCAL result\n"
  "    weight_GABAA = weight\n"
  "    weight_GABAB = weight*GABAB_ratio\n"
  "    : Locals:\n"
  "    : Psurv - survival probability of unrecovered state\n"
  "    : tsyn - time since last surival evaluation.\n"
  "\n"
  "\n"
  "    INITIAL{\n"
  "		tsyn=t\n"
  "    }\n"
  "\n"
  "    : Do not perform any calculations if the synapse (netcon) is deactivated.  This avoids drawing from the random stream\n"
  "    if(  !(weight > 0) ) {\n"
  "VERBATIM\n"
  "        return;\n"
  "ENDVERBATIM\n"
  "    }\n"
  "\n"
  "        : calc u at event-\n"
  "        if (Fac > 0) {\n"
  "                u = u*exp(-(t - tsyn_fac)/Fac) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.\n"
  "           } else {\n"
  "                  u = Use  \n"
  "           } \n"
  "           if(Fac > 0){\n"
  "                  u = u + Use*(1-u) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.\n"
  "           }    \n"
  "\n"
  "	   : tsyn_fac knows about all spikes, not only those that released\n"
  "	   : i.e. each spike can increase the u, regardless of recovered state.\n"
  "	   tsyn_fac = t\n"
  "\n"
  "           : recovery\n"
  "	   if (Rstate == 0) {\n"
  "	   : probability of survival of unrecovered state based on Poisson recovery with rate 1/tau\n"
  "	          Psurv = exp(-(t-tsyn)/Dep)\n"
  "		  result = urand()\n"
  "		  if (result>Psurv) {\n"
  "		         Rstate = 1     : recover      \n"
  "\n"
  "                         if( verboseLevel > 0 ) {\n"
  "                             printf( \"Recovered! %f at time %g: Psurv = %g, urand=%g\\n\", synapseID, t, Psurv, result )\n"
  "                         }\n"
  "\n"
  "		  }\n"
  "		  else {\n"
  "		         : survival must now be from this interval\n"
  "		         tsyn = t\n"
  "                         if( verboseLevel > 0 ) {\n"
  "                             printf( \"Failed to recover! %f at time %g: Psurv = %g, urand=%g\\n\", synapseID, t, Psurv, result )\n"
  "                         }\n"
  "		  }\n"
  "           }	   \n"
  "	   \n"
  "	   if (Rstate == 1) {\n"
  "   	          result = urand()\n"
  "		  if (result<u) {\n"
  "		  : release!\n"
  "   		         tsyn = t\n"
  "			 Rstate = 0\n"
  "\n"
  "                         A_GABAA = A_GABAA + weight_GABAA*factor_GABAA\n"
  "                         B_GABAA = B_GABAA + weight_GABAA*factor_GABAA\n"
  "                         A_GABAB = A_GABAB + weight_GABAB*factor_GABAB\n"
  "                         B_GABAB = B_GABAB + weight_GABAB*factor_GABAB\n"
  "                         \n"
  "                         if( verboseLevel > 0 ) {\n"
  "                             printf( \"Release! %f at time %g: vals %g %g %g \\n\", synapseID, t, A_GABAA, weight_GABAA, factor_GABAA )\n"
  "                         }\n"
  "		  		  \n"
  "		  }\n"
  "		  else {\n"
  "		         if( verboseLevel > 0 ) {\n"
  "			     printf(\"Failure! %f at time %g: urand = %g\\n\", synapseID, t, result )\n"
  "		         }\n"
  "\n"
  "		  }\n"
  "\n"
  "	   }\n"
  "\n"
  "        \n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "PROCEDURE setRNG() {\n"
  "VERBATIM\n"
  "    {\n"
  "        /**\n"
  "         * This function takes a NEURON Random object declared in hoc and makes it usable by this mod file.\n"
  "         * Note that this method is taken from Brett paper as used by netstim.hoc and netstim.mod\n"
  "         */\n"
  "        void** pv = (void**)(&_p_rng);\n"
  "        if( ifarg(1)) {\n"
  "            *pv = nrn_random_arg(1);\n"
  "        } else {\n"
  "            *pv = (void*)0;\n"
  "        }\n"
  "    }\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "FUNCTION urand() {\n"
  "VERBATIM\n"
  "        double value;\n"
  "        if (_p_rng) {\n"
  "                /*\n"
  "                :Supports separate independent but reproducible streams for\n"
  "                : each instance. However, the corresponding hoc Random\n"
  "                : distribution MUST be set to Random.uniform(1)\n"
  "                */\n"
  "                value = nrn_random_pick(_p_rng);\n"
  "                //printf(\"random stream for this simulation = %lf\\n\",value);\n"
  "                return value;\n"
  "        }else{\n"
  "ENDVERBATIM\n"
  "                : the old standby. Cannot use if reproducible parallel sim\n"
  "                : independent of nhost or which host this instance is on\n"
  "                : is desired, since each instance on this cpu draws from\n"
  "                : the same stream\n"
  "                urand = scop_random(1)\n"
  "VERBATIM\n"
  "        }\n"
  "ENDVERBATIM\n"
  "        urand = value\n"
  "}\n"
  "\n"
  "FUNCTION toggleVerbose() {\n"
  "    verboseLevel = 1 - verboseLevel\n"
  "}\n"
  ;
#endif
