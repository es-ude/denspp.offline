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
 
#define nrn_init _nrn_init__trel
#define _nrn_initial _nrn_initial__trel
#define nrn_cur _nrn_cur__trel
#define _nrn_current _nrn_current__trel
#define nrn_jacob _nrn_jacob__trel
#define nrn_state _nrn_state__trel
#define _net_receive _net_receive__trel 
#define release release__trel 
 
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
#define Ves _p[0]
#define Ves_columnindex 0
#define B _p[1]
#define B_columnindex 1
#define Ach _p[2]
#define Ach_columnindex 2
#define X _p[3]
#define X_columnindex 3
#define cai _p[4]
#define cai_columnindex 4
#define DVes _p[5]
#define DVes_columnindex 5
#define DB _p[6]
#define DB_columnindex 6
#define DAch _p[7]
#define DAch_columnindex 7
#define DX _p[8]
#define DX_columnindex 8
#define b _p[9]
#define b_columnindex 9
#define kt _p[10]
#define kt_columnindex 10
#define kpow _p[11]
#define kpow_columnindex 11
#define v _p[12]
#define v_columnindex 12
#define _g _p[13]
#define _g_columnindex 13
#define _ion_cai	*_ppvar[0]._pval
 
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
 /* declaration of user functions */
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
 "setdata_trel", _hoc_setdata,
 0, 0
};
 /* declare global and static user variables */
#define Aase Aase_trel
 double Aase = 4;
#define Arev Arev_trel
 double Arev = 0;
#define Agen Agen_trel
 double Agen = 4;
#define GenVes GenVes_trel
 double GenVes = 5;
#define Kd Kd_trel
 double Kd = 1e-12;
#define al al_trel
 double al = 141;
#define power power_trel
 double power = 2;
#define tauGen tauGen_trel
 double tauGen = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "Aase_trel", 0, 1e+09,
 "Arev_trel", 0, 1e+09,
 "Agen_trel", 0, 1e+09,
 "GenVes_trel", 0, 1e+09,
 "Kd_trel", 0, 1e+09,
 "al_trel", 0, 1e+09,
 "power_trel", 0, 10,
 "tauGen_trel", 1e-06, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "GenVes_trel", "mM",
 "tauGen_trel", "ms",
 "al_trel", "/mM2-ms",
 "Kd_trel", "mM2",
 "Agen_trel", "/ms",
 "Arev_trel", "/ms",
 "Aase_trel", "/ms",
 "Ves_trel", "mM",
 "B_trel", "mM",
 "Ach_trel", "mM",
 "X_trel", "mM",
 0,0
};
 static double Ach0 = 0;
 static double B0 = 0;
 static double Ves0 = 0;
 static double X0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "GenVes_trel", &GenVes_trel,
 "tauGen_trel", &tauGen_trel,
 "power_trel", &power_trel,
 "al_trel", &al_trel,
 "Kd_trel", &Kd_trel,
 "Agen_trel", &Agen_trel,
 "Arev_trel", &Arev_trel,
 "Aase_trel", &Aase_trel,
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
 
#define _cvode_ieq _ppvar[1]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"trel",
 0,
 0,
 "Ves_trel",
 "B_trel",
 "Ach_trel",
 "X_trel",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 14, _prop);
 	/*initialize range parameters*/
 	_prop->param = _p;
 	_prop->param_size = 14;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 2, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 
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

 void _release_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 3);
  _extcall_thread = (Datum*)ecalloc(2, sizeof(Datum));
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 14, 2);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 trel /root/nrn/build/cmake_install/share/nrn/demo/release/release.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "transmitter release";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 extern double *_nrn_thread_getelm(SparseObj*, int, int);
 
#define _MATELM1(_row,_col) *(_nrn_thread_getelm(_so, _row + 1, _col + 1))
 
#define _RHS1(_arg) _rhs[_arg+1]
  
#define _linmat1  1
 static int _spth1 = 1;
 static int _cvspth1 = 0;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[4], _dlist1[4]; static double *_temp1;
 static int release();
 
static int release (void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt)
 {int _reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<4;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 b = Kd * al ;
   if ( tauGen  == 0.0 ) {
     kt = 1e9 ;
     }
   else {
     kt = 1.0 / tauGen ;
     }
   kpow = al * pow( cai , power ) ;
   /* ~ GenVes <-> Ves ( kt , kt )*/
 f_flux =  kt * GenVes ;
 b_flux =  kt * Ves ;
 _RHS1( 2) += (f_flux - b_flux);
 
 _term =  kt ;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ Ves <-> B ( kpow , b )*/
 f_flux =  kpow * Ves ;
 b_flux =  b * B ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  kpow ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  b ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ B <-> Ach ( Agen , Arev )*/
 f_flux =  Agen * B ;
 b_flux =  Arev * Ach ;
 _RHS1( 1) -= (f_flux - b_flux);
 _RHS1( 0) += (f_flux - b_flux);
 
 _term =  Agen ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 0 ,1)  -= _term;
 _term =  Arev ;
 _MATELM1( 1 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ Ach <-> X ( Aase , 0.0 )*/
 f_flux =  Aase * Ach ;
 b_flux =  0.0 * X ;
 _RHS1( 0) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  Aase ;
 _MATELM1( 0 ,0)  += _term;
 _MATELM1( 3 ,0)  -= _term;
 _term =  0.0 ;
 _MATELM1( 0 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
    } return _reset;
 }
 
/*CVODE ode begin*/
 static int _ode_spec1(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<4;_i++) _p[_dlist1[_i]] = 0.0;}
 b = Kd * al ;
 if ( tauGen  == 0.0 ) {
   kt = 1e9 ;
   }
 else {
   kt = 1.0 / tauGen ;
   }
 kpow = al * pow( cai , power ) ;
 /* ~ GenVes <-> Ves ( kt , kt )*/
 f_flux =  kt * GenVes ;
 b_flux =  kt * Ves ;
 DVes += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ Ves <-> B ( kpow , b )*/
 f_flux =  kpow * Ves ;
 b_flux =  b * B ;
 DVes -= (f_flux - b_flux);
 DB += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ B <-> Ach ( Agen , Arev )*/
 f_flux =  Agen * B ;
 b_flux =  Arev * Ach ;
 DB -= (f_flux - b_flux);
 DAch += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ Ach <-> X ( Aase , 0.0 )*/
 f_flux =  Aase * Ach ;
 b_flux =  0.0 * X ;
 DAch -= (f_flux - b_flux);
 DX += (f_flux - b_flux);
 
 /*REACTION*/
    } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1(void* _so, double* _rhs, double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<4;_i++){
  	_RHS1(_i) = _dt1*(_p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 b = Kd * al ;
 if ( tauGen  == 0.0 ) {
 kt = 1e9 ;
 }
 else {
 kt = 1.0 / tauGen ;
 }
 kpow = al * pow( cai , power ) ;
 /* ~ GenVes <-> Ves ( kt , kt )*/
 _term =  kt ;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ Ves <-> B ( kpow , b )*/
 _term =  kpow ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 1 ,2)  -= _term;
 _term =  b ;
 _MATELM1( 2 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ B <-> Ach ( Agen , Arev )*/
 _term =  Agen ;
 _MATELM1( 1 ,1)  += _term;
 _MATELM1( 0 ,1)  -= _term;
 _term =  Arev ;
 _MATELM1( 1 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ Ach <-> X ( Aase , 0.0 )*/
 _term =  Aase ;
 _MATELM1( 0 ,0)  += _term;
 _MATELM1( 3 ,0)  -= _term;
 _term =  0.0 ;
 _MATELM1( 0 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
    } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 4;}
 
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
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 4; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _cvode_sparse_thread(&_thread[_cvspth1]._pvoid, 4, _dlist1, _p, _ode_matsol1, _ppvar, _thread, _nt);
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
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_cleanup(Datum* _thread) {
   _nrn_destroy_sparseobj_thread(_thread[_cvspth1]._pvoid);
   _nrn_destroy_sparseobj_thread(_thread[_spth1]._pvoid);
 }
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  Ach = Ach0;
  B = B0;
  Ves = Ves0;
  X = X0;
 {
   Ves = GenVes ;
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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{
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
  cai = _ion_cai;
 {  sparse_thread(&_thread[_spth1]._pvoid, 4, _slist1, _dlist1, _p, &t, dt, release, _linmat1, _ppvar, _thread, _nt);
     if (secondorder) {
    int _i;
    for (_i = 0; _i < 4; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 } {
   }
}}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = Ach_columnindex;  _dlist1[0] = DAch_columnindex;
 _slist1[1] = B_columnindex;  _dlist1[1] = DB_columnindex;
 _slist1[2] = Ves_columnindex;  _dlist1[2] = DVes_columnindex;
 _slist1[3] = X_columnindex;  _dlist1[3] = DX_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/root/nrn/build/cmake_install/share/nrn/demo/release/release.mod";
static const char* nmodl_file_text = 
  "TITLE transmitter release\n"
  ": taken from jwm simulation\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX trel\n"
  "	USEION ca READ cai\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(mM) = (milli/liter)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "\n"
  "	GenVes = 5	(mM)	<0,1e9>\n"
  "	tauGen = 0	(ms)	<1e-6,1e9>\n"
  "	power = 2		<0,10>\n"
  "	\n"
  "	al = 141	(/mM2-ms) <0,1e9>\n"
  "	Kd = 1e-12	(mM2)	<0,1e9>\n"
  "	Agen = 4	(/ms)	<0,1e9>\n"
  "	Arev = 0	(/ms)	<0,1e9>\n"
  "	Aase = 4	(/ms)	<0,1e9>\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	cai (mM)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	Ves	(mM)\n"
  "	B	(mM)\n"
  "	Ach	(mM)\n"
  "	X	(mM)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	Ves = GenVes\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE release METHOD sparse\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	b (/ms)\n"
  "	kt (/ms)\n"
  "	kpow (/ms)\n"
  "}\n"
  "\n"
  "KINETIC release {\n"
  "	b = Kd*al\n"
  "	if (tauGen == 0) {\n"
  "		kt=1e9\n"
  "	}else{\n"
  "		kt = 1/tauGen\n"
  "	}\n"
  "	kpow = al*cai^power	: set up so dimensionally correct when power=2\n"
  "\n"
  "	~ GenVes <-> Ves	(kt, kt)\n"
  "	~ Ves <-> B		(kpow, b)\n"
  "	~ B <-> Ach		(Agen, Arev)\n"
  "	~ Ach <-> X		(Aase, 0)\n"
  "}\n"
  ;
#endif
