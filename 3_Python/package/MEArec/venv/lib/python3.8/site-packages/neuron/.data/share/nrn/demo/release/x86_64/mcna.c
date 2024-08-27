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
 
#include "nmodlmutex.h" 
#define nrn_init _nrn_init__MCna
#define _nrn_initial _nrn_initial__MCna
#define nrn_cur _nrn_cur__MCna
#define _nrn_current _nrn_current__MCna
#define nrn_jacob _nrn_jacob__MCna
#define nrn_state _nrn_state__MCna
#define _net_receive _net_receive__MCna 
#define _f_rate _f_rate__MCna 
#define rate rate__MCna 
#define states states__MCna 
 
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
#define gnabar _p[0]
#define gnabar_columnindex 0
#define lp _p[1]
#define lp_columnindex 1
#define ml _p[2]
#define ml_columnindex 2
#define nm _p[3]
#define nm_columnindex 3
#define ina _p[4]
#define ina_columnindex 4
#define gna _p[5]
#define gna_columnindex 5
#define P _p[6]
#define P_columnindex 6
#define L _p[7]
#define L_columnindex 7
#define M _p[8]
#define M_columnindex 8
#define N _p[9]
#define N_columnindex 9
#define O _p[10]
#define O_columnindex 10
#define DP _p[11]
#define DP_columnindex 11
#define DL _p[12]
#define DL_columnindex 12
#define DM _p[13]
#define DM_columnindex 13
#define DN _p[14]
#define DN_columnindex 14
#define DO _p[15]
#define DO_columnindex 15
#define ena _p[16]
#define ena_columnindex 16
#define am _p[17]
#define am_columnindex 17
#define bm _p[18]
#define bm_columnindex 18
#define ah _p[19]
#define ah_columnindex 19
#define bh _p[20]
#define bh_columnindex 20
#define v _p[21]
#define v_columnindex 21
#define _g _p[22]
#define _g_columnindex 22
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
 
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
 static void _hoc_alp(void);
 static void _hoc_bet(void);
 static void _hoc_expM1(void);
 static void _hoc_rate(void);
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
 "setdata_MCna", _hoc_setdata,
 "alp_MCna", _hoc_alp,
 "bet_MCna", _hoc_bet,
 "expM1_MCna", _hoc_expM1,
 "rate_MCna", _hoc_rate,
 0, 0
};
#define alp alp_MCna
#define bet bet_MCna
#define expM1 expM1_MCna
 extern double alp( _threadargsprotocomma_ double , double );
 extern double bet( _threadargsprotocomma_ double , double );
 extern double expM1( _threadargsprotocomma_ double , double );
 
static void _check_rate(double*, Datum*, Datum*, NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type) {
   _check_rate(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define cnt2 cnt2_MCna
 double cnt2 = 0;
#define cnt1 cnt1_MCna
 double cnt1 = 0;
#define usetable usetable_MCna
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "gnabar_MCna", 0, 1e+09,
 "lp_MCna", 0, 1e+09,
 "ml_MCna", 0, 1e+09,
 "nm_MCna", 0, 1e+09,
 "usetable_MCna", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gnabar_MCna", "mho/cm2",
 "ina_MCna", "mA/cm2",
 "gna_MCna", "mho/cm2",
 0,0
};
 static double L0 = 0;
 static double M0 = 0;
 static double N0 = 0;
 static double O0 = 0;
 static double P0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "cnt1_MCna", &cnt1_MCna,
 "cnt2_MCna", &cnt2_MCna,
 "usetable_MCna", &usetable_MCna,
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
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"MCna",
 "gnabar_MCna",
 "lp_MCna",
 "ml_MCna",
 "nm_MCna",
 0,
 "ina_MCna",
 "gna_MCna",
 0,
 "P_MCna",
 "L_MCna",
 "M_MCna",
 "N_MCna",
 "O_MCna",
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 23, _prop);
 	/*initialize range parameters*/
 	gnabar = 0.12;
 	lp = 1.9;
 	ml = 0.75;
 	nm = 0.3;
 	_prop->param = _p;
 	_prop->param_size = 23;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _thread_cleanup(Datum*);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _mcna_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 3);
  _extcall_thread = (Datum*)ecalloc(2, sizeof(Datum));
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 23, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 MCna /root/nrn/build/cmake_install/share/nrn/demo/release/mcna.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_am;
 static double *_t_ah;
 static double *_t_bm;
 static double *_t_bh;
static int _reset;
static char *modelname = "Moore-Cox sodium channel";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_rate(_threadargsprotocomma_ double);
static int rate(_threadargsprotocomma_ double);
 
#define _MATELM1(_row,_col) *(_nrn_thread_getelm(_so, _row + 1, _col + 1))
 
#define _RHS1(_arg) _rhs[_arg+1]
  static int _cvspth1 = 1;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 extern double *_nrn_thread_getelm(SparseObj*, int, int);
 
#define _MATELM1(_row,_col) *(_nrn_thread_getelm(_so, _row + 1, _col + 1))
 
#define _RHS1(_arg) _rhs[_arg+1]
  
#define _linmat1  1
 static int _spth1 = 0;
 static void _n_rate(_threadargsprotocomma_ double _lv);
 static int _slist1[5], _dlist1[5]; static double *_temp1;
 static int states();
 
static int states (void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt)
 {int _reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<5;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 /* PROTECT */_NMODLMUTEXLOCK
 cnt2 = cnt2 + 1.0 ;
   
 _NMODLMUTEXUNLOCK /* end PROTECT */
 rate ( _threadargscomma_ v * 1.0 ) ;
   /* ~ P <-> L ( am , lp * bm )*/
 f_flux =  am * P ;
 b_flux =  lp * bm * L ;
 _RHS1( 4) -= (f_flux - b_flux);
 _RHS1( 0) += (f_flux - b_flux);
 
 _term =  am ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 0 ,4)  -= _term;
 _term =  lp * bm ;
 _MATELM1( 4 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ L <-> M ( 2.0 * am , ml * bm )*/
 f_flux =  2.0 * am * L ;
 b_flux =  ml * bm * M ;
 _RHS1( 0) -= (f_flux - b_flux);
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  2.0 * am ;
 _MATELM1( 0 ,0)  += _term;
 _MATELM1( 1 ,0)  -= _term;
 _term =  ml * bm ;
 _MATELM1( 0 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ M <-> N ( 3.0 * am , nm * bm )*/
 f_flux =  3.0 * am * M ;
 b_flux =  nm * bm * N ;
 _RHS1( 1) -= (f_flux - b_flux);
 _RHS1( 2) += (f_flux - b_flux);
 
 _term =  3.0 * am ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 2 ,1)  -= _term;
 _term =  nm * bm ;
 _MATELM1( 1 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ N <-> O ( 1.1 * bh , 0.0 )*/
 f_flux =  1.1 * bh * N ;
 b_flux =  0.0 * O ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  1.1 * bh ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 3 ,2)  -= _term;
 _term =  0.0 ;
 _MATELM1( 2 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ N <-> P ( 3.0 * bm , 0.0 )*/
 f_flux =  3.0 * bm * N ;
 b_flux =  0.0 * P ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 4) += (f_flux - b_flux);
 
 _term =  3.0 * bm ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 4 ,2)  -= _term;
 _term =  0.0 ;
 _MATELM1( 2 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ P <-> O ( bh , ah )*/
 f_flux =  bh * P ;
 b_flux =  ah * O ;
 _RHS1( 4) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  bh ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  ah ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
    } return _reset;
 }
 
double alp ( _threadargsprotocomma_ double _lv , double _li ) {
   double _lalp;
 double _la , _lb , _lc , _lq10 ;
 _lv = - _lv - 65.0 ;
   _lq10 = pow( 3.0 , ( ( celsius - 6.3 ) / 10.0 ) ) ;
   if ( _li  == 0.0 ) {
     _lalp = _lq10 * .1 * expM1 ( _threadargscomma_ _lv + 25.0 , 10.0 ) ;
     }
   else if ( _li  == 1.0 ) {
     _lalp = _lq10 * .07 * exp ( _lv / 20.0 ) ;
     }
   
return _lalp;
 }
 
static void _hoc_alp(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alp ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
double bet ( _threadargsprotocomma_ double _lv , double _li ) {
   double _lbet;
 double _la , _lb , _lc , _lq10 ;
 _lv = - _lv - 65.0 ;
   _lq10 = pow( 3.0 , ( ( celsius - 6.3 ) / 10.0 ) ) ;
   if ( _li  == 0.0 ) {
     _lbet = _lq10 * 4.0 * exp ( _lv / 18.0 ) ;
     }
   else if ( _li  == 1.0 ) {
     _lbet = _lq10 * 1.0 / ( exp ( .1 * _lv + 3.0 ) + 1.0 ) ;
     }
   
return _lbet;
 }
 
static void _hoc_bet(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  bet ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
double expM1 ( _threadargsprotocomma_ double _lx , double _ly ) {
   double _lexpM1;
 if ( fabs ( _lx / _ly ) < 1e-6 ) {
     _lexpM1 = _ly * ( 1.0 - _lx / _ly / 2.0 ) ;
     }
   else {
     _lexpM1 = _lx / ( exp ( _lx / _ly ) - 1.0 ) ;
     }
   
return _lexpM1;
 }
 
static void _hoc_expM1(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  expM1 ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 static double _mfac_rate, _tmin_rate;
  static void _check_rate(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_celsius;
  if (!usetable) {return;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_rate =  - 100.0 ;
   _tmax =  100.0 ;
   _dx = (_tmax - _tmin_rate)/200.; _mfac_rate = 1./_dx;
   for (_i=0, _x=_tmin_rate; _i < 201; _x += _dx, _i++) {
    _f_rate(_p, _ppvar, _thread, _nt, _x);
    _t_am[_i] = am;
    _t_ah[_i] = ah;
    _t_bm[_i] = bm;
    _t_bh[_i] = bh;
   }
   _sav_celsius = celsius;
  }
 }

 static int rate(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv) { 
#if 0
_check_rate(_p, _ppvar, _thread, _nt);
#endif
 _n_rate(_p, _ppvar, _thread, _nt, _lv);
 return 0;
 }

 static void _n_rate(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_rate(_p, _ppvar, _thread, _nt, _lv); return; 
}
 _xi = _mfac_rate * (_lv - _tmin_rate);
 if (isnan(_xi)) {
  am = _xi;
  ah = _xi;
  bm = _xi;
  bh = _xi;
  return;
 }
 if (_xi <= 0.) {
 am = _t_am[0];
 ah = _t_ah[0];
 bm = _t_bm[0];
 bh = _t_bh[0];
 return; }
 if (_xi >= 200.) {
 am = _t_am[200];
 ah = _t_ah[200];
 bm = _t_bm[200];
 bh = _t_bh[200];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 am = _t_am[_i] + _theta*(_t_am[_i+1] - _t_am[_i]);
 ah = _t_ah[_i] + _theta*(_t_ah[_i+1] - _t_ah[_i]);
 bm = _t_bm[_i] + _theta*(_t_bm[_i+1] - _t_bm[_i]);
 bh = _t_bh[_i] + _theta*(_t_bh[_i+1] - _t_bh[_i]);
 }

 
static int  _f_rate ( _threadargsprotocomma_ double _lv ) {
   double _la , _lb , _ltau ;
 am = alp ( _threadargscomma_ _lv , 0.0 ) ;
   ah = alp ( _threadargscomma_ _lv , 1.0 ) ;
   bm = bet ( _threadargscomma_ _lv , 0.0 ) ;
   bh = bet ( _threadargscomma_ _lv , 1.0 ) ;
    return 0; }
 
static void _hoc_rate(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_rate(_p, _ppvar, _thread, _nt);
#endif
 _r = 1.;
 rate ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
/*CVODE ode begin*/
 static int _ode_spec1(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<5;_i++) _p[_dlist1[_i]] = 0.0;}
 /* PROTECT */_NMODLMUTEXLOCK
 cnt2 = cnt2 + 1.0 ;
 
 _NMODLMUTEXUNLOCK /* end PROTECT */
 rate ( _threadargscomma_ v * 1.0 ) ;
 /* ~ P <-> L ( am , lp * bm )*/
 f_flux =  am * P ;
 b_flux =  lp * bm * L ;
 DP -= (f_flux - b_flux);
 DL += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ L <-> M ( 2.0 * am , ml * bm )*/
 f_flux =  2.0 * am * L ;
 b_flux =  ml * bm * M ;
 DL -= (f_flux - b_flux);
 DM += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ M <-> N ( 3.0 * am , nm * bm )*/
 f_flux =  3.0 * am * M ;
 b_flux =  nm * bm * N ;
 DM -= (f_flux - b_flux);
 DN += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ N <-> O ( 1.1 * bh , 0.0 )*/
 f_flux =  1.1 * bh * N ;
 b_flux =  0.0 * O ;
 DN -= (f_flux - b_flux);
 DO += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ N <-> P ( 3.0 * bm , 0.0 )*/
 f_flux =  3.0 * bm * N ;
 b_flux =  0.0 * P ;
 DN -= (f_flux - b_flux);
 DP += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ P <-> O ( bh , ah )*/
 f_flux =  bh * P ;
 b_flux =  ah * O ;
 DP -= (f_flux - b_flux);
 DO += (f_flux - b_flux);
 
 /*REACTION*/
    } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1(void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<5;_i++){
  	_RHS1(_i) = _dt1*(_p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 /* PROTECT */_NMODLMUTEXLOCK
 cnt2 = cnt2 + 1.0 ;
 
 _NMODLMUTEXUNLOCK /* end PROTECT */
 rate ( _threadargscomma_ v * 1.0 ) ;
 /* ~ P <-> L ( am , lp * bm )*/
 _term =  am ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 0 ,4)  -= _term;
 _term =  lp * bm ;
 _MATELM1( 4 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ L <-> M ( 2.0 * am , ml * bm )*/
 _term =  2.0 * am ;
 _MATELM1( 0 ,0)  += _term;
 _MATELM1( 1 ,0)  -= _term;
 _term =  ml * bm ;
 _MATELM1( 0 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ M <-> N ( 3.0 * am , nm * bm )*/
 _term =  3.0 * am ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 2 ,1)  -= _term;
 _term =  nm * bm ;
 _MATELM1( 1 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ N <-> O ( 1.1 * bh , 0.0 )*/
 _term =  1.1 * bh ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 3 ,2)  -= _term;
 _term =  0.0 ;
 _MATELM1( 2 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ N <-> P ( 3.0 * bm , 0.0 )*/
 _term =  3.0 * bm ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 4 ,2)  -= _term;
 _term =  0.0 ;
 _MATELM1( 2 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ P <-> O ( bh , ah )*/
 _term =  bh ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  ah ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
    } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 5;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 5; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _cvode_sparse_thread(&_thread[_cvspth1]._pvoid, 5, _dlist1, _p, _ode_matsol1, _ppvar, _thread, _nt);
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
  ena = _ion_ena;
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_cleanup(Datum* _thread) {
   _nrn_destroy_sparseobj_thread(_thread[_spth1]._pvoid);
   _nrn_destroy_sparseobj_thread(_thread[_cvspth1]._pvoid);
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  L = L0;
  M = M0;
  N = N0;
  O = O0;
  P = P0;
 {
   cnt1 = 0.0 ;
   cnt2 = 0.0 ;
   P = 1.0 ;
   rate ( _threadargscomma_ v * 1.0 ) ;
    _ss_sparse_thread(&_thread[_spth1]._pvoid, 5, _slist1, _dlist1, _p, &t, dt, states, _linmat1, _ppvar, _thread, _nt);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 5; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
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
 _check_rate(_p, _ppvar, _thread, _nt);
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
  ena = _ion_ena;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ina = gnabar * N * ( v - ena ) ;
   /* PROTECT */_NMODLMUTEXLOCK
 cnt1 = cnt1 + 1.0 ;
   
 _NMODLMUTEXUNLOCK /* end PROTECT */
 }
 _current += ina;

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
  ena = _ion_ena;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
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
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
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
  ena = _ion_ena;
 {  sparse_thread(&_thread[_spth1]._pvoid, 5, _slist1, _dlist1, _p, &t, dt, states, _linmat1, _ppvar, _thread, _nt);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 5; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 } }}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
   _t_am = makevector(201*sizeof(double));
   _t_ah = makevector(201*sizeof(double));
   _t_bm = makevector(201*sizeof(double));
   _t_bh = makevector(201*sizeof(double));
 _slist1[0] = L_columnindex;  _dlist1[0] = DL_columnindex;
 _slist1[1] = M_columnindex;  _dlist1[1] = DM_columnindex;
 _slist1[2] = N_columnindex;  _dlist1[2] = DN_columnindex;
 _slist1[3] = O_columnindex;  _dlist1[3] = DO_columnindex;
 _slist1[4] = P_columnindex;  _dlist1[4] = DP_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/root/nrn/build/cmake_install/share/nrn/demo/release/mcna.mod";
static const char* nmodl_file_text = 
  "TITLE Moore-Cox sodium channel\n"
  ": Biophy. J. (1976) 16:171\n"
  ": This paper mapped HH VClamp currents, no attention paid to resting currents. \n"
  ": Used V-jump @ T=0 to check match with HH action  potential; fig shows equal V-threshold levels\n"
  ": Ramon first noted instability at rest, spontaneous impulse generation.\n"
  ": Same problem noted with resimulation with NEURON. Now thresholds rather different\n"
  ": Revised July 28, 1995 to remove instability. Added back reaction rate coefficients for HH beta m\n"
  ": First use of  NEURON's new \"Run Fitter\" to find best values of these coefficients,\n"
  ": using delay and fitting  ina to just beyond  peak.\n"
  ": Excellent fit to HH AP with these coefficients except for \"gratituitous hump\" in HH\n"
  ": Changed from HH reference potential level at rest to NEURON @-65mV\n"
  "\n"
  "NEURON {\n"
  "     SUFFIX MCna\n"
  "     USEION na READ ena WRITE ina\n"
  "     RANGE gnabar, gna, ina, lp, ml, nm, porate     \n"
  "                 :lp, ml, nm, porate coefficients of HH betas for re-fitting\n"
  "	GLOBAL cnt1, cnt2\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "     (mA) = (milliamp)\n"
  "     (mV) = (millivolt)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "     gnabar=.120 (mho/cm2)	<0,1e9>\n"
  "                lp=1.9		<0, 1e9>\n"
  "                ml=.75		<0, 1e9>\n"
  "                nm=.3		<0, 1e9>\n"
  ":                porate=1\n"
  "}\n"
  "STATE {\n"
  "     P L M N O\n"
  "}\n"
  "ASSIGNED {\n"
  "     v (mV)\n"
  "     celsius (degC) : 6.3 \n"
  "     ena (mV)\n"
  "     cnt1 cnt2\n"
  "     ina (mA/cm2)\n"
  "     gna (mho/cm2)\n"
  "}\n"
  "\n"
  "ASSIGNED {  am (/ms)   bm (/ms)  ah (/ms)  bh (/ms)}\n"
  "\n"
  "INITIAL {\n"
  "	cnt1 = 0\n"
  "	cnt2 = 0\n"
  "     P=1\n"
  "     rate(v*1(/mV))\n"
  "     SOLVE states STEADYSTATE sparse\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "     SOLVE states METHOD sparse\n"
  "     ina = gnabar*N*(v - ena)\n"
  "	PROTECT cnt1 = cnt1 + 1\n"
  "}\n"
  "\n"
  "KINETIC states {\n"
  "	PROTECT cnt2 = cnt2 + 1\n"
  "     rate(v*1(/mV))\n"
  ":     CONSERVE P + L + M + N + O = 1\n"
  "     ~ P <-> L (am, lp*bm)    :back reaction in original = 3.5   \n"
  "     ~ L <-> M (2*am, ml*bm)          :back reaction in original = 0\n"
  "     ~ M <-> N (3*am, nm*bm)         :back reaction in original = 0\n"
  "     ~ N <-> O (1.1*bh, 0)\n"
  "     ~ N <-> P (3*bm, 0)\n"
  "     ~ P <-> O (bh, ah)       :back reaction in original = 1,\n"
  "  			      : found this to still be good\n"
  "}\n"
  "\n"
  "UNITSOFF\n"
  "FUNCTION alp(v(mV),i) { LOCAL a,b,c,q10 :rest = -65  order m,h\n"
  "     v = -v - 65  :convert to hh convention\n"
  "     q10 = 3^((celsius - 6.3)/10)\n"
  "     if (i==0) {\n"
  "          alp = q10*.1*expM1(v + 25, 10)\n"
  "     }else if (i==1){\n"
  "          alp = q10*.07*exp(v/20)\n"
  "     }\n"
  "}\n"
  "\n"
  "FUNCTION bet(v,i) { LOCAL a,b,c,q10 :rest = -70  order m,h\n"
  "     v = -v - 65\n"
  "     q10 = 3^((celsius - 6.3)/10)\n"
  "     if (i==0) {\n"
  "          bet = q10* 4*exp(v/18)\n"
  "     }else if (i==1){\n"
  "          bet = q10*1/(exp(.1*v + 3) + 1)\n"
  "     }\n"
  "}\n"
  "\n"
  "FUNCTION expM1(x,y) {\n"
  "     if (fabs(x/y) < 1e-6) {\n"
  "          expM1 = y*(1 - x/y/2) : for singular point\n"
  "     }else{\n"
  "          expM1 = x/(exp(x/y) - 1)\n"
  "     }\n"
  "}\n"
  "\n"
  "PROCEDURE rate(v) {LOCAL a, b, tau :rest = -65\n"
  "     TABLE am, ah, bm, bh DEPEND celsius FROM -100 TO 100 WITH 200\n"
  "     am = alp(v, 0)\n"
  "     ah = alp(v, 1)\n"
  "     bm = bet(v, 0)\n"
  "     bh = bet(v, 1)\n"
  "}\n"
  "UNITSON\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ;
#endif
