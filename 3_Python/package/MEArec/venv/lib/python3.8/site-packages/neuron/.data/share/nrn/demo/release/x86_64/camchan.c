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
 
#define nrn_init _nrn_init__cachan
#define _nrn_initial _nrn_initial__cachan
#define nrn_cur _nrn_cur__cachan
#define _nrn_current _nrn_current__cachan
#define nrn_jacob _nrn_jacob__cachan
#define nrn_state _nrn_state__cachan
#define _net_receive _net_receive__cachan 
#define castate castate__cachan 
 
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
#define pcabar _p[0]
#define pcabar_columnindex 0
#define ica _p[1]
#define ica_columnindex 1
#define oca _p[2]
#define oca_columnindex 2
#define cai _p[3]
#define cai_columnindex 3
#define cao _p[4]
#define cao_columnindex 4
#define Doca _p[5]
#define Doca_columnindex 5
#define v _p[6]
#define v_columnindex 6
#define _g _p[7]
#define _g_columnindex 7
#define _ion_cai	*_ppvar[0]._pval
#define _ion_cao	*_ppvar[1]._pval
#define _ion_ica	*_ppvar[2]._pval
#define _ion_dicadv	*_ppvar[3]._pval
 
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
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_efun(void);
 static void _hoc_ghk(void);
 static void _hoc_oca_tau(void);
 static void _hoc_oca_ss(void);
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

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_cachan", _hoc_setdata,
 "efun_cachan", _hoc_efun,
 "ghk_cachan", _hoc_ghk,
 "oca_tau_cachan", _hoc_oca_tau,
 "oca_ss_cachan", _hoc_oca_ss,
 0, 0
};
#define _f_oca_tau _f_oca_tau_cachan
#define _f_oca_ss _f_oca_ss_cachan
#define efun efun_cachan
#define ghk ghk_cachan
#define oca_tau oca_tau_cachan
#define oca_ss oca_ss_cachan
 extern double _f_oca_tau( _threadargsprotocomma_ double );
 extern double _f_oca_ss( _threadargsprotocomma_ double );
 extern double efun( _threadargsprotocomma_ double );
 extern double ghk( _threadargsprotocomma_ double , double , double );
 extern double oca_tau( _threadargsprotocomma_ double );
 extern double oca_ss( _threadargsprotocomma_ double );
 
static void _check_oca_ss(double*, Datum*, Datum*, NrnThread*); 
static void _check_oca_tau(double*, Datum*, Datum*, NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type) {
   _check_oca_ss(_p, _ppvar, _thread, _nt);
   _check_oca_tau(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define taufactor taufactor_cachan
 double taufactor = 2;
#define usetable usetable_cachan
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "pcabar_cachan", 0, 1e+09,
 "taufactor_cachan", 1e-06, 1e+06,
 "usetable_cachan", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "taufactor_cachan", "1e-6",
 "pcabar_cachan", "cm/s",
 "ica_cachan", "mA/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double oca0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "taufactor_cachan", &taufactor_cachan,
 "usetable_cachan", &usetable_cachan,
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
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[4]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"cachan",
 "pcabar_cachan",
 0,
 "ica_cachan",
 0,
 "oca_cachan",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 8, _prop);
 	/*initialize range parameters*/
 	pcabar = 2e-08;
 	_prop->param = _p;
 	_prop->param_size = 8;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[1]._pval = &prop_ion->param[2]; /* cao */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _camchan_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 8, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 cachan /root/nrn/build/cmake_install/share/nrn/demo/release/camchan.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define FARADAY _nrnunit_FARADAY[_nrnunit_use_legacy_]
static double _nrnunit_FARADAY[2] = {0x1.78e555060882cp+16, 96485.3}; /* 96485.3321233100141 */
 
#define R _nrnunit_R[_nrnunit_use_legacy_]
static double _nrnunit_R[2] = {0x1.0a1013e8990bep+3, 8.3145}; /* 8.3144626181532395 */
 static double *_t_oca_ss;
 static double *_t_oca_tau;
static int _reset;
static char *modelname = "CaChan";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static double _n_oca_tau(_threadargsprotocomma_ double _lv);
 static double _n_oca_ss(_threadargsprotocomma_ double _lv);
 static int _slist1[1], _dlist1[1];
 static int castate(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   double _linf , _ltau ;
 _linf = oca_ss ( _threadargscomma_ v ) ;
   _ltau = oca_tau ( _threadargscomma_ v ) ;
   Doca = ( _linf - oca ) / _ltau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 double _linf , _ltau ;
 _linf = oca_ss ( _threadargscomma_ v ) ;
 _ltau = oca_tau ( _threadargscomma_ v ) ;
 Doca = Doca  / (1. - dt*( ( ( ( - 1.0 ) ) ) / _ltau )) ;
  return 0;
}
 /*END CVODE*/
 static int castate (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   double _linf , _ltau ;
 _linf = oca_ss ( _threadargscomma_ v ) ;
   _ltau = oca_tau ( _threadargscomma_ v ) ;
    oca = oca + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / _ltau)))*(- ( ( ( _linf ) ) / _ltau ) / ( ( ( ( - 1.0 ) ) ) / _ltau ) - oca) ;
   }
  return 0;
}
 
double ghk ( _threadargsprotocomma_ double _lv , double _lci , double _lco ) {
   double _lghk;
 double _lz , _leci , _leco ;
 _lz = ( 1e-3 ) * 2.0 * FARADAY * _lv / ( R * ( celsius + 273.15 ) ) ;
   _leco = _lco * efun ( _threadargscomma_ _lz ) ;
   _leci = _lci * efun ( _threadargscomma_ - _lz ) ;
   _lghk = ( .001 ) * 2.0 * FARADAY * ( _leci - _leco ) ;
   
return _lghk;
 }
 
static void _hoc_ghk(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ghk ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) );
 hoc_retpushx(_r);
}
 
double efun ( _threadargsprotocomma_ double _lz ) {
   double _lefun;
 if ( fabs ( _lz ) < 1e-4 ) {
     _lefun = 1.0 - _lz / 2.0 ;
     }
   else {
     _lefun = _lz / ( exp ( _lz ) - 1.0 ) ;
     }
   
return _lefun;
 }
 
static void _hoc_efun(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  efun ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 static double _mfac_oca_ss, _tmin_oca_ss;
  static void _check_oca_ss(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_oca_ss =  - 150.0 ;
   _tmax =  150.0 ;
   _dx = (_tmax - _tmin_oca_ss)/200.; _mfac_oca_ss = 1./_dx;
   for (_i=0, _x=_tmin_oca_ss; _i < 201; _x += _dx, _i++) {
    _t_oca_ss[_i] = _f_oca_ss(_p, _ppvar, _thread, _nt, _x);
   }
  }
 }

 double oca_ss(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv) { 
#if 0
_check_oca_ss(_p, _ppvar, _thread, _nt);
#endif
 return _n_oca_ss(_p, _ppvar, _thread, _nt, _lv);
 }

 static double _n_oca_ss(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_oca_ss(_p, _ppvar, _thread, _nt, _lv); 
}
 _xi = _mfac_oca_ss * (_lv - _tmin_oca_ss);
 if (isnan(_xi)) {
  return _xi; }
 if (_xi <= 0.) {
 return _t_oca_ss[0];
 }
 if (_xi >= 200.) {
 return _t_oca_ss[200];
 }
 _i = (int) _xi;
 return _t_oca_ss[_i] + (_xi - (double)_i)*(_t_oca_ss[_i+1] - _t_oca_ss[_i]);
 }

 
double _f_oca_ss ( _threadargsprotocomma_ double _lv ) {
   double _loca_ss;
 double _la , _lb ;
 _lv = _lv + 65.0 ;
   _la = 1.0 * efun ( _threadargscomma_ .1 * ( 25.0 - _lv ) ) ;
   _lb = 4.0 * exp ( - _lv / 18.0 ) ;
   _loca_ss = _la / ( _la + _lb ) ;
   
return _loca_ss;
 }
 
static void _hoc_oca_ss(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_oca_ss(_p, _ppvar, _thread, _nt);
#endif
 _r =  oca_ss ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 static double _mfac_oca_tau, _tmin_oca_tau;
  static void _check_oca_tau(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_celsius;
  static double _sav_taufactor;
  if (!usetable) {return;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_sav_taufactor != taufactor) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_oca_tau =  - 150.0 ;
   _tmax =  150.0 ;
   _dx = (_tmax - _tmin_oca_tau)/200.; _mfac_oca_tau = 1./_dx;
   for (_i=0, _x=_tmin_oca_tau; _i < 201; _x += _dx, _i++) {
    _t_oca_tau[_i] = _f_oca_tau(_p, _ppvar, _thread, _nt, _x);
   }
   _sav_celsius = celsius;
   _sav_taufactor = taufactor;
  }
 }

 double oca_tau(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv) { 
#if 0
_check_oca_tau(_p, _ppvar, _thread, _nt);
#endif
 return _n_oca_tau(_p, _ppvar, _thread, _nt, _lv);
 }

 static double _n_oca_tau(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_oca_tau(_p, _ppvar, _thread, _nt, _lv); 
}
 _xi = _mfac_oca_tau * (_lv - _tmin_oca_tau);
 if (isnan(_xi)) {
  return _xi; }
 if (_xi <= 0.) {
 return _t_oca_tau[0];
 }
 if (_xi >= 200.) {
 return _t_oca_tau[200];
 }
 _i = (int) _xi;
 return _t_oca_tau[_i] + (_xi - (double)_i)*(_t_oca_tau[_i+1] - _t_oca_tau[_i]);
 }

 
double _f_oca_tau ( _threadargsprotocomma_ double _lv ) {
   double _loca_tau;
 double _la , _lb , _lq ;
 _lq = pow( 3.0 , ( ( celsius - 6.3 ) / 10.0 ) ) ;
   _lv = _lv + 65.0 ;
   _la = 1.0 * efun ( _threadargscomma_ .1 * ( 25.0 - _lv ) ) ;
   _lb = 4.0 * exp ( - _lv / 18.0 ) ;
   _loca_tau = taufactor / ( _la + _lb ) ;
   
return _loca_tau;
 }
 
static void _hoc_oca_tau(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_oca_tau(_p, _ppvar, _thread, _nt);
#endif
 _r =  oca_tau ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 1;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 1; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 2);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  oca = oca0;
 {
   oca = oca_ss ( _threadargscomma_ v ) ;
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

#if 0
 _check_oca_ss(_p, _ppvar, _thread, _nt);
 _check_oca_tau(_p, _ppvar, _thread, _nt);
#endif
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
  cai = _ion_cai;
  cao = _ion_cao;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ica = pcabar * oca * oca * ghk ( _threadargscomma_ v , cai , cao ) ;
   }
 _current += ica;

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
  cai = _ion_cai;
  cao = _ion_cao;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dica;
  _dica = ica;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dicadv += (_dica - ica)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica ;
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
  cai = _ion_cai;
  cao = _ion_cao;
 {   castate(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = oca_columnindex;  _dlist1[0] = Doca_columnindex;
   _t_oca_ss = makevector(201*sizeof(double));
   _t_oca_tau = makevector(201*sizeof(double));
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/root/nrn/build/cmake_install/share/nrn/demo/release/camchan.mod";
static const char* nmodl_file_text = 
  "TITLE CaChan\n"
  ": Calcium Channel with Goldman- Hodgkin-Katz permeability\n"
  ": The fraction of open calcium channels has the same kinetics as\n"
  ":   the HH m process but is slower by taufactor\n"
  "\n"
  "UNITS {\n"
  "	(molar) = (1/liter)\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mV) =	(millivolt)\n"
  "	(mA) =	(milliamp)\n"
  "	(mM) =	(millimolar)\n"
  "}\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX cachan\n"
  "	USEION ca READ cai, cao WRITE ica\n"
  "	RANGE pcabar, ica\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	:FARADAY = 96520 (coul)\n"
  "	:R = 8.3134 (joule/degC)\n"
  "	FARADAY = (faraday) (coulomb)\n"
  "	R = (k-mole) (joule/degC)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	taufactor=2	<1e-6, 1e6>: Time constant factor relative to standard HH\n"
  "	pcabar=.2e-7	(cm/s)	<0, 1e9>: Maximum Permeability\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	celsius		(degC)\n"
  "	v		(mV)\n"
  "	cai		(mM)\n"
  "	cao		(mM)\n"
  "	ica		(mA/cm2)\n"
  "}\n"
  "\n"
  "STATE {	oca }		: fraction of open channels\n"
  "\n"
  "INITIAL {\n"
  "	oca = oca_ss(v)\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE castate METHOD cnexp\n"
  "	ica = pcabar*oca*oca*ghk(v, cai, cao)\n"
  "}\n"
  "\n"
  "DERIVATIVE castate {\n"
  "	LOCAL inf, tau\n"
  "	inf = oca_ss(v)  tau = oca_tau(v)\n"
  "	oca' = (inf - oca)/tau\n"
  "}\n"
  "\n"
  "FUNCTION ghk(v(mV), ci(mM), co(mM)) (.001 coul/cm3) {\n"
  "	LOCAL z, eci, eco\n"
  "	z = (1e-3)*2*FARADAY*v/(R*(celsius+273.15))\n"
  "	eco = co*efun(z)\n"
  "	eci = ci*efun(-z)\n"
  "	:high cao charge moves inward\n"
  "	:negative potential charge moves inward\n"
  "	ghk = (.001)*2*FARADAY*(eci - eco)\n"
  "}\n"
  "\n"
  "FUNCTION efun(z) {\n"
  "	if (fabs(z) < 1e-4) {\n"
  "		efun = 1 - z/2\n"
  "	}else{\n"
  "		efun = z/(exp(z) - 1)\n"
  "	}\n"
  "}\n"
  "\n"
  "FUNCTION oca_ss(v(mV)) {\n"
  "	LOCAL a, b\n"
  "	TABLE FROM -150 TO 150 WITH 200\n"
  "	\n"
  "	v = v+65\n"
  "	a = 1(1/ms)*efun(.1(1/mV)*(25-v))\n"
  "	b = 4(1/ms)*exp(-v/18(mV))\n"
  "	oca_ss = a/(a + b)\n"
  "}\n"
  "\n"
  "FUNCTION oca_tau(v(mV)) (ms) {\n"
  "	LOCAL a, b, q\n"
  "	TABLE DEPEND celsius, taufactor FROM -150 TO 150 WITH 200\n"
  "\n"
  "	q = 3^((celsius - 6.3)/10 (degC))\n"
  "	v = v+65\n"
  "	a = 1(1/ms)*efun(.1(1/mV)*(25-v))\n"
  "	b = 4(1/ms)*exp(-v/18(mV))\n"
  "	oca_tau = taufactor/(a + b)\n"
  "}\n"
  "\n"
  "COMMENT\n"
  "This model is related to the passive model in that it also describes\n"
  "a membrane channel. However it involves two new concepts in that the\n"
  "channel is ion selective and the conductance of the channel is\n"
  "described by a state variable.\n"
  "\n"
  "Since many membrane mechanisms involve specific ions whose concentration\n"
  "governs a channel current (either directly or via a Nernst potential) and since\n"
  "the sum of the ionic currents of these mechanisms in turn may govern\n"
  "the concentration, it is necessary that NEURON be explicitly told which\n"
  "ionic variables are being used by this model and which are being computed.\n"
  "This is done by the USEION statement.  This statement uses the indicated\n"
  "base name for an ion (call it `base') and ensures the existance of\n"
  "four range variables that can be used by any mechanism that requests them\n"
  "via the USEION statement. I.e. these variables are shared by the different\n"
  "mechanisms.  The four variables are the current, ibase; the\n"
  "equilibrium potential, ebase; the internal concentration, basei; and the\n"
  "external concentration, baseo. (Note that Ca and ca would be distinct\n"
  "ion species).  The READ part of the statement lists the subset of these\n"
  "four variables which are needed as input to the this model's computations.\n"
  "Any changes to those variables within this mechanism will be lost on exit.\n"
  "The WRITE part of the statement lists the subset which are computed by\n"
  "the present mechanism.  If the current is computed, then it's value\n"
  "on exit will be added to the neuron wide value of ibase and will also\n"
  "be added to the total membrane current that is used to calculate the\n"
  "membrane potential.\n"
  "\n"
  "When this model is `insert'ed, fcurrent() executes all the statements\n"
  "of the EQUATION block EXCEPT the SOLVE statement. I.e. the states are\n"
  "NOT integrated in time.  The fadvance() function executes the entire\n"
  "EQUATION block including the SOLVE statement; thus the states are integrated\n"
  "over the interval t to t+dt.\n"
  "\n"
  "Notice that several mechanisms can WRITE to ibase; but it is an error\n"
  "if several mechanisms (in the same section) WRITE to ebase, baseo, or basei.\n"
  "\n"
  "This model makes use of several variables known specially to NEURON. They are\n"
  "celsius, v, and t.  It implicitly makes use of dt.\n"
  "\n"
  "TABLE refers to a special type of FUNCTION in which the value of the\n"
  "function is computed by table lookup with linear interpolation of\n"
  "the table entries.  TABLE's are recomputed automatically whenever a\n"
  "variable that the table depends on (Through the DEPEND list; not needed\n"
  "in these tables) is changed.\n"
  "The TABLE statement indicates the minimum and maximum values of the argument\n"
  "and the number of table entries.  From NEURON, the function oca_ss_cachan(v)\n"
  "returns the proper value in the table. When the variable \"usetable_cachan\"\n"
  "is set to 0, oca_ss_cachan(v)returns the true function value.\n"
  "Thus the table error can be easily plotted.\n"
  "ENDCOMMENT\n"
  ;
#endif
