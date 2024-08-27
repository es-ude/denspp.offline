/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
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
 
#define nrn_init _nrn_init__cadifpmp
#define _nrn_initial _nrn_initial__cadifpmp
#define nrn_cur _nrn_cur__cadifpmp
#define _nrn_current _nrn_current__cadifpmp
#define nrn_jacob _nrn_jacob__cadifpmp
#define nrn_state _nrn_state__cadifpmp
#define _net_receive _net_receive__cadifpmp 
#define coord coord__cadifpmp 
#define parms parms__cadifpmp 
#define state state__cadifpmp 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define ica_pmp _p[0]
#define ica_pmp_columnindex 0
#define ica_pmp_last _p[1]
#define ica_pmp_last_columnindex 1
#define ca (_p + 2)
#define ca_columnindex 2
#define pump _p[12]
#define pump_columnindex 12
#define pumpca _p[13]
#define pumpca_columnindex 13
#define cao _p[14]
#define cao_columnindex 14
#define cai _p[15]
#define cai_columnindex 15
#define ica _p[16]
#define ica_columnindex 16
#define area1 _p[17]
#define area1_columnindex 17
#define c1 _p[18]
#define c1_columnindex 18
#define c2 _p[19]
#define c2_columnindex 19
#define c3 _p[20]
#define c3_columnindex 20
#define c4 _p[21]
#define c4_columnindex 21
#define Dca (_p + 22)
#define Dca_columnindex 22
#define Dpump _p[32]
#define Dpump_columnindex 32
#define Dpumpca _p[33]
#define Dpumpca_columnindex 33
#define _g _p[34]
#define _g_columnindex 34
#define _ion_cao	*_ppvar[0]._pval
#define _ion_ica	*_ppvar[1]._pval
#define _ion_cai	*_ppvar[2]._pval
#define _ion_dicadv	*_ppvar[3]._pval
#define _style_ca	*((int*)_ppvar[4]._pvoid)
#define diam	*_ppvar[5]._pval
 
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
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_coord(void);
 static void _hoc_parms(void);
 static void _hoc_ss(void);
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
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_cadifpmp", _hoc_setdata,
 "coord_cadifpmp", _hoc_coord,
 "parms_cadifpmp", _hoc_parms,
 "ss_cadifpmp", _hoc_ss,
 0, 0
};
#define ss ss_cadifpmp
 extern double ss( );
 /* declare global and static user variables */
#define DFree DFree_cadifpmp
 double DFree = 0.6;
#define beta beta_cadifpmp
 double beta = 50;
#define k4 k4_cadifpmp
 double k4 = 5;
#define k3 k3_cadifpmp
 double k3 = 500;
#define k2 k2_cadifpmp
 double k2 = 250000;
#define k1 k1_cadifpmp
 double k1 = 5e+08;
#define pump0 pump0_cadifpmp
 double pump0 = 3e-14;
#define vol vol_cadifpmp
 double vol[10];
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "DFree_cadifpmp", 0, 1e+09,
 "beta_cadifpmp", 0, 1e+09,
 "k4_cadifpmp", 0, 1e+10,
 "k3_cadifpmp", 0, 1e+10,
 "k2_cadifpmp", 0, 1e+10,
 "k1_cadifpmp", 0, 1e+10,
 "pump0_cadifpmp", 0, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "DFree_cadifpmp", "um2/ms",
 "k1_cadifpmp", "/mM-s",
 "k2_cadifpmp", "/s",
 "k3_cadifpmp", "/s",
 "k4_cadifpmp", "/mM-s",
 "pump0_cadifpmp", "mol/cm2",
 "vol_cadifpmp", "1",
 "ca_cadifpmp", "mM",
 "pump_cadifpmp", "mol/cm2",
 "pumpca_cadifpmp", "mol/cm2",
 "ica_pmp_cadifpmp", "mA/cm2",
 "ica_pmp_last_cadifpmp", "mA/cm2",
 0,0
};
 static double ca0 = 0;
 static double delta_t = 0.01;
 static double pumpca0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "DFree_cadifpmp", &DFree_cadifpmp,
 "beta_cadifpmp", &beta_cadifpmp,
 "k1_cadifpmp", &k1_cadifpmp,
 "k2_cadifpmp", &k2_cadifpmp,
 "k3_cadifpmp", &k3_cadifpmp,
 "k4_cadifpmp", &k4_cadifpmp,
 "pump0_cadifpmp", &pump0_cadifpmp,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 "vol_cadifpmp", vol_cadifpmp, 10,
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
 
#define _cvode_ieq _ppvar[6]._i
 static void _ode_synonym(int, double**, Datum**);
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"cadifpmp",
 0,
 "ica_pmp_cadifpmp",
 "ica_pmp_last_cadifpmp",
 0,
 "ca_cadifpmp[10]",
 "pump_cadifpmp",
 "pumpca_cadifpmp",
 0,
 0};
 static Symbol* _morphology_sym;
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 35, _prop);
 	/*initialize range parameters*/
 	_prop->param = _p;
 	_prop->param_size = 35;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 7, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_morphology_sym);
 	_ppvar[5]._pval = &prop_ion->param[0]; /* diam */
 prop_ion = need_memb(_ca_sym);
 nrn_check_conc_write(_prop, prop_ion, 1);
 nrn_promote(prop_ion, 3, 0);
 	_ppvar[0]._pval = &prop_ion->param[2]; /* cao */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[2]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 	_ppvar[4]._pvoid = (void*)(&(prop_ion->dparam[0]._i)); /* iontype for ca */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 "ca_cadifpmp", 1e-06,
 "pump_cadifpmp", 1e-15,
 "pumpca_cadifpmp", 1e-15,
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _cabpump_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("ca", -10000.);
 	_morphology_sym = hoc_lookup("morphology");
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 35, 7);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "#ca_ion");
  hoc_register_dparam_semantics(_mechtype, 6, "cvodeieq");
  hoc_register_dparam_semantics(_mechtype, 5, "diam");
 	nrn_writes_conc(_mechtype, 0);
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_synonym(_mechtype, _ode_synonym);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 cadifpmp /root/nrn/build/cmake_install/share/nrn/demo/release/cabpump.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define FARADAY _nrnunit_FARADAY[_nrnunit_use_legacy_]
static double _nrnunit_FARADAY[2] = {0x1.78e555060882cp+16, 96485.3}; /* 96485.3321233100141 */
 
#define PI _nrnunit_PI[_nrnunit_use_legacy_]
static double _nrnunit_PI[2] = {0x1.921fb54442d18p+1, 3.14159}; /* 3.14159265358979312 */
 
#define R _nrnunit_R[_nrnunit_use_legacy_]
static double _nrnunit_R[2] = {0x1.0a1013e8990bep+3, 8.3145}; /* 8.3144626181532395 */
 static double volo = 1;
 static double _zfrat [ 10 ] ;
static int _reset;
static char *modelname = "Calcium ion accumulation and diffusion with pump";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int coord();
static int parms();
 extern double *_getelm();
 
#define _MATELM1(_row,_col)	*(_getelm(_row + 1, _col + 1))
 
#define _RHS1(_arg) _coef1[_arg + 1]
 static double *_coef1;
 
#define _linmat1  0
 static void* _sparseobj1;
 static void* _cvsparseobj1;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[12], _dlist1[12]; static double *_temp1;
 static int state();
 
static int  coord (  ) {
   double _lr , _ldr2 ;
 _lr = 1.0 / 2.0 ;
   _ldr2 = _lr / ( 10.0 - 1.0 ) / 2.0 ;
   vol [ 0 ] = 0.0 ;
   _zfrat [ 0 ] = 2.0 * _lr ;
   {int  _li ;for ( _li = 0 ; _li <= 10 - 2 ; _li ++ ) {
     vol [ _li ] = vol [ _li ] + PI * ( _lr - _ldr2 / 2.0 ) * 2.0 * _ldr2 ;
     _lr = _lr - _ldr2 ;
     _zfrat [ _li + 1 ] = 2.0 * PI * _lr / ( 2.0 * _ldr2 ) ;
     _lr = _lr - _ldr2 ;
     vol [ _li + 1 ] = PI * ( _lr + _ldr2 / 2.0 ) * 2.0 * _ldr2 ;
     } }
    return 0; }
 
static void _hoc_coord(void) {
  double _r;
   _r = 1.;
 coord (  );
 hoc_retpushx(_r);
}
 
static int state ()
 {_reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<12;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
}  
_RHS1(10) *= ( ( 1e10 ) * area1) ;
_MATELM1(10, 10) *= ( ( 1e10 ) * area1); 
_RHS1(11) *= ( ( 1e10 ) * area1) ;
_MATELM1(11, 11) *= ( ( 1e10 ) * area1);  
for (_i=0; _i < 10; _i++) {
  	_RHS1(_i + 0) *= ( ( 1.0 + beta ) * diam * diam * vol [ ((int) _i ) ] * 1.0) ;
_MATELM1(_i + 0, _i + 0) *= ( ( 1.0 + beta ) * diam * diam * vol [ ((int) _i ) ] * 1.0);  } }
 /* COMPARTMENT _li , ( 1.0 + beta ) * diam * diam * vol [ ((int) _i ) ] * 1.0 {
     ca }
   */
 /* COMPARTMENT ( 1e10 ) * area1 {
     pump pumpca }
   */
 /* COMPARTMENT volo * ( 1e15 ) {
     }
   */
 /* ~ pumpca <-> pump + cao ( c3 , c4 )*/
 f_flux =  c3 * pumpca ;
 b_flux =  c4 * cao * pump ;
 _RHS1( 10) -= (f_flux - b_flux);
 _RHS1( 11) += (f_flux - b_flux);
 
 _term =  c3 ;
 _MATELM1( 10 ,10)  += _term;
 _MATELM1( 11 ,10)  -= _term;
 _term =  c4 * cao ;
 _MATELM1( 10 ,11)  -= _term;
 _MATELM1( 11 ,11)  += _term;
 /*REACTION*/
  ica_pmp = ( 1e-4 ) * 2.0 * FARADAY * ( f_flux - b_flux ) / area1 ;
   /* ~ ca [ 0 ] < < ( - ( ica - ica_pmp_last ) * PI * diam * 1.0 * ( 1e4 ) * _zfrat [ 0 ] / ( 2.0 * FARADAY ) )*/
 f_flux = b_flux = 0.;
 _RHS1( 0 +  0) += (b_flux =   ( - ( ica - ica_pmp_last ) * PI * diam * 1.0 * ( 1e4 ) * _zfrat [ 0 ] / ( 2.0 * FARADAY ) ) );
 /*FLUX*/
  {int  _li ;for ( _li = 0 ; _li <= 10 - 2 ; _li ++ ) {
     /* ~ ca [ _li ] <-> ca [ _li + 1 ] ( DFree * _zfrat [ _li + 1 ] * 1.0 , DFree * _zfrat [ _li + 1 ] * 1.0 )*/
 f_flux =  DFree * _zfrat [ _li + 1 ] * 1.0 * ca [ _li] ;
 b_flux =  DFree * _zfrat [ _li + 1 ] * 1.0 * ca [ _li + 1] ;
 _RHS1( 0 +  _li) -= (f_flux - b_flux);
 _RHS1( 0 +  _li + 1) += (f_flux - b_flux);
 
 _term =  DFree * _zfrat [ _li + 1 ] * 1.0 ;
 _MATELM1( 0 +  _li ,0 +  _li)  += _term;
 _MATELM1( 0 +  _li + 1 ,0 +  _li)  -= _term;
 _term =  DFree * _zfrat [ _li + 1 ] * 1.0 ;
 _MATELM1( 0 +  _li ,0 +  _li + 1)  -= _term;
 _MATELM1( 0 +  _li + 1 ,0 +  _li + 1)  += _term;
 /*REACTION*/
  } }
   /* ~ ca [ 0 ] + pump <-> pumpca ( c1 , c2 )*/
 f_flux =  c1 * pump * ca [ 0] ;
 b_flux =  c2 * pumpca ;
 _RHS1( 11) -= (f_flux - b_flux);
 _RHS1( 0 +  0) -= (f_flux - b_flux);
 _RHS1( 10) += (f_flux - b_flux);
 
 _term =  c1 * ca [ 0] ;
 _MATELM1( 11 ,11)  += _term;
 _MATELM1( 0 +  0 ,11)  += _term;
 _MATELM1( 10 ,11)  -= _term;
 _term =  c1 * pump ;
 _MATELM1( 11 ,0 +  0)  += _term;
 _MATELM1( 0 +  0 ,0 +  0)  += _term;
 _MATELM1( 10 ,0 +  0)  -= _term;
 _term =  c2 ;
 _MATELM1( 11 ,10)  -= _term;
 _MATELM1( 0 +  0 ,10)  -= _term;
 _MATELM1( 10 ,10)  += _term;
 /*REACTION*/
  cai = ca [ 0 ] ;
     } return _reset;
 }
 
static int  parms (  ) {
   coord ( _threadargs_ ) ;
   area1 = 2.0 * PI * ( diam / 2.0 ) * 1.0 ;
   c1 = ( 1e7 ) * area1 * k1 ;
   c2 = ( 1e7 ) * area1 * k2 ;
   c3 = ( 1e7 ) * area1 * k3 ;
   c4 = ( 1e7 ) * area1 * k4 ;
    return 0; }
 
static void _hoc_parms(void) {
  double _r;
   _r = 1.;
 parms (  );
 hoc_retpushx(_r);
}
 
double ss (  ) {
   double _lss;
 error = _ss_sparse(&_sparseobj1, 12, _slist1, _dlist1, _p, &t, dt, state,&_coef1, _linmat1);
 if(error){fprintf(stderr,"at line 147 in file cabpump.mod:\n	SOLVE state STEADYSTATE sparse\n"); nrn_complain(_p); abort_run(error);}
    if (secondorder) {
    int _i;
    for (_i = 0; _i < 12; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 _lss = cai ;
   
return _lss;
 }
 
static void _hoc_ss(void) {
  double _r;
   _r =  ss (  );
 hoc_retpushx(_r);
}
 
/*CVODE ode begin*/
 static int _ode_spec1() {_reset=0;{
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<12;_i++) _p[_dlist1[_i]] = 0.0;}
 /* COMPARTMENT _li , ( 1.0 + beta ) * diam * diam * vol [ ((int) _i ) ] * 1.0 {
   ca }
 */
 /* COMPARTMENT ( 1e10 ) * area1 {
   pump pumpca }
 */
 /* COMPARTMENT volo * ( 1e15 ) {
   }
 */
 /* ~ pumpca <-> pump + cao ( c3 , c4 )*/
 f_flux =  c3 * pumpca ;
 b_flux =  c4 * cao * pump ;
 Dpumpca -= (f_flux - b_flux);
 Dpump += (f_flux - b_flux);
 
 /*REACTION*/
  ica_pmp = ( 1e-4 ) * 2.0 * FARADAY * ( f_flux - b_flux ) / area1 ;
 /* ~ ca [ 0 ] < < ( - ( ica - ica_pmp_last ) * PI * diam * 1.0 * ( 1e4 ) * _zfrat [ 0 ] / ( 2.0 * FARADAY ) )*/
 f_flux = b_flux = 0.;
 Dca [ 0] += (b_flux =   ( - ( ica - ica_pmp_last ) * PI * diam * 1.0 * ( 1e4 ) * _zfrat [ 0 ] / ( 2.0 * FARADAY ) ) );
 /*FLUX*/
  {int  _li ;for ( _li = 0 ; _li <= 10 - 2 ; _li ++ ) {
   /* ~ ca [ _li ] <-> ca [ _li + 1 ] ( DFree * _zfrat [ _li + 1 ] * 1.0 , DFree * _zfrat [ _li + 1 ] * 1.0 )*/
 f_flux =  DFree * _zfrat [ _li + 1 ] * 1.0 * ca [ _li] ;
 b_flux =  DFree * _zfrat [ _li + 1 ] * 1.0 * ca [ _li + 1] ;
 Dca [ _li] -= (f_flux - b_flux);
 Dca [ _li + 1] += (f_flux - b_flux);
 
 /*REACTION*/
  } }
 /* ~ ca [ 0 ] + pump <-> pumpca ( c1 , c2 )*/
 f_flux =  c1 * pump * ca [ 0] ;
 b_flux =  c2 * pumpca ;
 Dpump -= (f_flux - b_flux);
 Dca [ 0] -= (f_flux - b_flux);
 Dpumpca += (f_flux - b_flux);
 
 /*REACTION*/
  cai = ca [ 0 ] ;
 for (_i=0; _i < 10; _i++) { _p[_dlist1[_i + 0]] /= ( ( 1.0 + beta ) * diam * diam * vol [ ((int) _i ) ] * 1.0);}
 _p[_dlist1[10]] /= ( ( 1e10 ) * area1);
 _p[_dlist1[11]] /= ( ( 1e10 ) * area1);
   } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1() {_reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<12;_i++){
  	_RHS1(_i) = _dt1*(_p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
}  
_RHS1(10) *= ( ( 1e10 ) * area1) ;
_MATELM1(10, 10) *= ( ( 1e10 ) * area1); 
_RHS1(11) *= ( ( 1e10 ) * area1) ;
_MATELM1(11, 11) *= ( ( 1e10 ) * area1);  
for (_i=0; _i < 10; _i++) {
  	_RHS1(_i + 0) *= ( ( 1.0 + beta ) * diam * diam * vol [ ((int) _i ) ] * 1.0) ;
_MATELM1(_i + 0, _i + 0) *= ( ( 1.0 + beta ) * diam * diam * vol [ ((int) _i ) ] * 1.0);  } }
 /* COMPARTMENT _li , ( 1.0 + beta ) * diam * diam * vol [ ((int) _i ) ] * 1.0 {
 ca }
 */
 /* COMPARTMENT ( 1e10 ) * area1 {
 pump pumpca }
 */
 /* COMPARTMENT volo * ( 1e15 ) {
 }
 */
 /* ~ pumpca <-> pump + cao ( c3 , c4 )*/
 _term =  c3 ;
 _MATELM1( 10 ,10)  += _term;
 _MATELM1( 11 ,10)  -= _term;
 _term =  c4 * cao ;
 _MATELM1( 10 ,11)  -= _term;
 _MATELM1( 11 ,11)  += _term;
 /* ~ ca [ 0 ] < < ( - ( ica - ica_pmp_last ) * PI * diam * 1.0 * ( 1e4 ) * _zfrat [ 0 ] / ( 2.0 * FARADAY ) )*/
 /*FLUX*/
  {int  _li ;for ( _li = 0 ; _li <= 10 - 2 ; _li ++ ) {
 /* ~ ca [ _li ] <-> ca [ _li + 1 ] ( DFree * _zfrat [ _li + 1 ] * 1.0 , DFree * _zfrat [ _li + 1 ] * 1.0 )*/
 _term =  DFree * _zfrat [ _li + 1 ] * 1.0 ;
 _MATELM1( 0 +  _li ,0 +  _li)  += _term;
 _MATELM1( 0 +  _li + 1 ,0 +  _li)  -= _term;
 _term =  DFree * _zfrat [ _li + 1 ] * 1.0 ;
 _MATELM1( 0 +  _li ,0 +  _li + 1)  -= _term;
 _MATELM1( 0 +  _li + 1 ,0 +  _li + 1)  += _term;
 /*REACTION*/
  } }
 /* ~ ca [ 0 ] + pump <-> pumpca ( c1 , c2 )*/
 _term =  c1 * ca [ 0] ;
 _MATELM1( 11 ,11)  += _term;
 _MATELM1( 0 +  0 ,11)  += _term;
 _MATELM1( 10 ,11)  -= _term;
 _term =  c1 * pump ;
 _MATELM1( 11 ,0 +  0)  += _term;
 _MATELM1( 0 +  0 ,0 +  0)  += _term;
 _MATELM1( 10 ,0 +  0)  -= _term;
 _term =  c2 ;
 _MATELM1( 11 ,10)  -= _term;
 _MATELM1( 0 +  0 ,10)  -= _term;
 _MATELM1( 10 ,10)  += _term;
 /*REACTION*/
  cai = ca [ 0 ] ;
   } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 12;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cao = _ion_cao;
  ica = _ion_ica;
  cai = _ion_cai;
     _ode_spec1 ();
  _ion_cai = cai;
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 12; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 static void _ode_synonym(int _cnt, double** _pp, Datum** _ppd) { 
 	int _i; 
	for (_i=0; _i < _cnt; ++_i) {_p = _pp[_i]; _ppvar = _ppd[_i];
 _ion_cai =  ca [ 0 ] ;
 }}
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _cvode_sparse(&_cvsparseobj1, 12, _dlist1, _p, _ode_matsol1, &_coef1);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cao = _ion_cao;
  ica = _ion_ica;
  cai = _ion_cai;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 2);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 1);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
 for (_i=0; _i<10; _i++) ca[_i] = ca0;
  pumpca = pumpca0;
  pump = pump0;
 {
   double _ltotal ;
 parms ( _threadargs_ ) ;
   {int  _li ;for ( _li = 0 ; _li <= 10 - 1 ; _li ++ ) {
     ca [ _li ] = cai ;
     } }
   pumpca = cai * pump * c1 / c2 ;
   _ltotal = pumpca + pump ;
   if ( _ltotal > 1e-9 ) {
     pump = pump * ( pump / _ltotal ) ;
     pumpca = pumpca * ( pump / _ltotal ) ;
     }
   ica_pmp = 0.0 ;
   ica_pmp_last = 0.0 ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
  cao = _ion_cao;
  ica = _ion_ica;
  cai = _ion_cai;
 initmodel();
  _ion_cai = cai;
   nrn_wrote_conc(_ca_sym, (&(_ion_cai)) - 1, _style_ca);
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   ica_pmp_last = ica_pmp ;
   ica = ica_pmp ;
   }
 _current += ica;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
  cao = _ion_cao;
  ica = _ion_ica;
  cai = _ion_cai;
if (_nt->_vcv) { _ode_spec1(); }
 _g = _nrn_current(_v + .001);
 	{ double _dica;
  _dica = ica;
 _rhs = _nrn_current(_v);
  _ion_dicadv += (_dica - ica)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_cai = cai;
  _ion_ica += ica ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 
}}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
double _dtsav = dt;
if (secondorder) { dt *= 0.5; }
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
  cao = _ion_cao;
  ica = _ion_ica;
  cai = _ion_cai;
 { error = sparse(&_sparseobj1, 12, _slist1, _dlist1, _p, &t, dt, state,&_coef1, _linmat1);
 if(error){fprintf(stderr,"at line 86 in file cabpump.mod:\n	SOLVE state METHOD sparse\n"); nrn_complain(_p); abort_run(error);}
    if (secondorder) {
    int _i;
    for (_i = 0; _i < 12; ++_i) {
      _p[_slist1[_i]] += dt*_p[_dlist1[_i]];
    }}
 }  _ion_cai = cai;
 }}
 dt = _dtsav;
}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 for(_i=0;_i<10;_i++){_slist1[0+_i] = ca_columnindex + _i;  _dlist1[0+_i] = Dca_columnindex + _i;}
 _slist1[10] = pumpca_columnindex;  _dlist1[10] = Dpumpca_columnindex;
 _slist1[11] = pump_columnindex;  _dlist1[11] = Dpump_columnindex;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/root/nrn/build/cmake_install/share/nrn/demo/release/cabpump.mod";
static const char* nmodl_file_text = 
  "TITLE Calcium ion accumulation and diffusion with pump\n"
  ": The internal coordinate system is set up in PROCEDURE coord_cadifus()\n"
  ": and must be executed before computing the concentrations.\n"
  ": The scale factors set up in this procedure do not have to be recomputed\n"
  ": when diam or DFree are changed.\n"
  ": The amount of calcium in an annulus is ca[i]*diam^2*vol[i] with\n"
  ": ca[0] being the second order correct concentration at the exact edge\n"
  ": and ca[NANN-1] being the concentration at the exact center\n"
  "\n"
  "? interface\n"
  "NEURON {\n"
  "	SUFFIX cadifpmp\n"
  "	USEION ca READ cao, ica WRITE cai, ica\n"
  "	RANGE ica_pmp, ica_pmp_last\n"
  "	GLOBAL vol, pump0\n"
  "}\n"
  "\n"
  "DEFINE NANN  10\n"
  "\n"
  "UNITS {\n"
  "	(mV)	= (millivolt)\n"
  "	(molar) = (1/liter)\n"
  "	(mM)	= (millimolar)\n"
  "	(um)	= (micron)\n"
  "	(mA)	= (milliamp)\n"
  "	(mol)	= (1)\n"
  "	FARADAY = (faraday)	 (coulomb)\n"
  "	PI	= (pi)		(1)\n"
  "	R 	= (k-mole)	(joule/degC)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	DFree = .6	(um2/ms) <0,1e9>\n"
  "	beta = 50		<0, 1e9>\n"
  "\n"
  "        k1 = 5e8        (/mM-s) <0, 1e10>:optional mm formulation\n"
  "        k2 = .25e6      (/s)	<0, 1e10>\n"
  "        k3 = .5e3       (/s)	<0, 1e10>\n"
  "        k4 = 5e0        (/mM-s)	<0, 1e10>\n"
  "	pump0 = 3e-14 (mol/cm2) <0, 1e9> : set to 0 in hoc if this pump not wanted\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	celsius		(degC)\n"
  "	diam		(um)\n"
  "	v		(millivolt)\n"
  "	cao		(mM)\n"
  "	cai		(mM)\n"
  "	ica		(mA/cm2)\n"
  "	vol[NANN]	(1)	: gets extra cm2 when multiplied by diam^2\n"
  "        ica_pmp (mA/cm2)\n"
  "        area1    (um2)\n"
  "        c1      (1+8 um5/ms)\n"
  "        c2      (1-10 um2/ms)\n"
  "        c3      (1-10 um2/ms)\n"
  "        c4      (1+8 um5/ms)\n"
  "	ica_pmp_last (mA/cm2)\n"
  "}\n"
  "\n"
  "CONSTANT {\n"
  "	volo = 1 (liter)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	ca[NANN]	(mM) <1e-6> : ca[0] is equivalent to cai\n"
  "        pump    (mol/cm2)	<1e-15>\n"
  "        pumpca  (mol/cm2)	<1e-15>\n"
  "}\n"
  "\n"
  "INITIAL {LOCAL total\n"
  "	parms()\n"
  "	FROM i=0 TO NANN-1 {\n"
  "		ca[i] = cai\n"
  "	}\n"
  "	pumpca = cai*pump*c1/c2\n"
  "	total = pumpca + pump\n"
  "	if (total > 1e-9) {\n"
  "		pump = pump*(pump/total)\n"
  "		pumpca = pumpca*(pump/total)\n"
  "	}\n"
  "	ica_pmp = 0\n"
  "	ica_pmp_last = 0\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE state METHOD sparse\n"
  "	ica_pmp_last = ica_pmp\n"
  "	ica = ica_pmp\n"
  ":	printf(\"Breakpoint t=%g v=%g cai=%g ica=%g\\n\", t, v, cai, ica)\n"
  "}\n"
  "\n"
  "LOCAL frat[NANN] 	: gets extra cm when multiplied by diam\n"
  "\n"
  "PROCEDURE coord() {\n"
  "	LOCAL r, dr2\n"
  "	: cylindrical coordinate system  with constant annuli thickness to\n"
  "	: center of cell. Note however that the first annulus is half thickness\n"
  "	: so that the concentration is second order correct spatially at\n"
  "	: the membrane or exact edge of the cell.\n"
  "	: note ca[0] is at edge of cell\n"
  "	:      ca[NANN-1] is at center of cell\n"
  "	r = 1/2					:starts at edge (half diam)\n"
  "	dr2 = r/(NANN-1)/2			:half thickness of annulus\n"
  "	vol[0] = 0\n"
  "	frat[0] = 2*r\n"
  "	FROM i=0 TO NANN-2 {\n"
  "		vol[i] = vol[i] + PI*(r-dr2/2)*2*dr2	:interior half\n"
  "		r = r - dr2\n"
  "		frat[i+1] = 2*PI*r/(2*dr2)	:exterior edge of annulus\n"
  "					: divided by distance between centers\n"
  "		r = r - dr2\n"
  "		vol[i+1] = PI*(r+dr2/2)*2*dr2	:outer half of annulus\n"
  "	}\n"
  "}\n"
  "\n"
  "KINETIC state {\n"
  ":	printf(\"Solve begin t=%g v=%g cai=%g ica_pmp=%g\\n\", t, v, cai, ica_pmp)\n"
  "	COMPARTMENT i, (1+beta)*diam*diam*vol[i]*1(um) {ca}\n"
  "	COMPARTMENT (1e10)*area1 {pump pumpca}\n"
  "	COMPARTMENT volo*(1e15) {cao}\n"
  "? kinetics\n"
  "	~ pumpca <-> pump + cao		(c3, c4)\n"
  "	ica_pmp = (1e-4)*2*FARADAY*(f_flux - b_flux)/area1\n"
  "	: all currents except pump\n"
  "	~ ca[0] << (-(ica-ica_pmp_last)*PI*diam*1(um)*(1e4)*frat[0]/(2*FARADAY))\n"
  "	:diffusion\n"
  "	FROM i=0 TO NANN-2 {\n"
  "		~ ca[i] <-> ca[i+1] (DFree*frat[i+1]*1(um), DFree*frat[i+1]*1(um))\n"
  "	}\n"
  "	:pump\n"
  "	~ ca[0] + pump <-> pumpca	(c1, c2)\n"
  "	cai = ca[0] : this assignment statement is used specially by cvode\n"
  ":	printf(\"Solve end cai=%g ica=%g ica_pmp=%g ica_pmp_last=%g\\n\",\n"
  ":		 cai, ica, ica_pmp,ica_pmp_last)\n"
  "}\n"
  "	\n"
  "PROCEDURE parms() {\n"
  "	coord()\n"
  "	area1 = 2*PI*(diam/2) * 1(um)\n"
  "        c1 = (1e7)*area1 * k1\n"
  "        c2 = (1e7)*area1 * k2\n"
  "        c3 = (1e7)*area1 * k3\n"
  "        c4 = (1e7)*area1 * k4  \n"
  "}\n"
  "\n"
  "FUNCTION ss() (mM) {\n"
  "	SOLVE state STEADYSTATE sparse\n"
  "	ss = cai\n"
  "}\n"
  "\n"
  "COMMENT\n"
  "At this time, conductances (and channel states and currents are\n"
  "calculated at the midpoint of a dt interval.  Membrane potential and\n"
  "concentrations are calculated at the edges of a dt interval.  With\n"
  "secondorder=2 everything turns out to be second order correct.\n"
  "ENDCOMMENT\n"
  ;
#endif
