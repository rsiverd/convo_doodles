/****************************************************************************/
/*                                                                          */
/*    Test routines for delta function (pixel basis) convolution kernel     */
/* fitting. Hopefully it works ...                                          */
/*                                                                          */
/* Rob Siverd                                                               */
/* Created:       2018-11-06                                                */
/* Last modified: 2018-11-06                                                */
/*                                                                          */
/****************************************************************************/
/****************************************************************************/
/*--------------------------------------------------------------------------*/

#include   <math.h>
#include  <stdio.h>
#include <stdlib.h>

/* Shared routines: */
#include  "imageIO.h"
#include "simpleIO.h"
#include "m_invert.h"

/* OpenMP multi-threading: */
#ifdef _OPENMP
   #include      <omp.h>
#endif

/* Useful min/max macro definitions: */
#ifndef MIN
#  define MIN(a,b) ((a) > (b) ? (b) : (a))
#endif
#ifndef MAX
#  define MAX(a,b) ((a) < (b) ? (b) : (a))
#endif

/*--------------------------------------------------------------------------*/
/* Test code for direct (delta function) unweighted kernel fitting. This    */
/* routine performs a least-squares fit to find the MxN kernel that best    */
/* maps src_stamp --> dst_stamp.                                            */
/*                                                                          */
/* NOTE/WARNING: this test implementation uses the ENTIRE IMAGE to fit the  */
/* convolution kernel. The fit may take a while ...                         */
int convo_ufit_img_quick (
                  ezImg  *src_stamp,   /*  [in] source/ref image structure  */
                  ezImg  *dst_stamp,   /*  [in] target/out image structure  */
                   long   halfx,       /* [par] X half-size of convo kernel */
                   long   halfy,       /* [par] Y half-size of convo kernel */
                  ezImg  *kern_vals,   /* [out] best-fit convolution kernel */
                  DTYPE  *background   /* [out] measured background offset? */
              //   long   xbuffer,     /* [par] : best-fit model parameters */
              // double **fitpar,      /* output: best-fit model parameters */
              //    int  *nparam       /* output: size of parameter vector  */
                     )
#define FUNCNAME "convo_ufit_img_quick"                     
{

   /* Three valid ezImg pointers required: */
   if ( !src_stamp || !dst_stamp || !kern_vals ) {
      fprintf(stderr, 
            "%s: received NULL pointer(s) in ezImg structures!\n"
            "src_stamp: %p\n"
            "dst_stamp: %p\n"
            "kern_vals: %p\n",
            FUNCNAME, src_stamp, dst_stamp, kern_vals);
      return 1; /* failure */
   }

   /* Variables and short-hand: */
   long i, j, X, Y;
   long SNX = (src_stamp->naxes)[0];
   long SNY = (src_stamp->naxes)[1];
   long DNX = (dst_stamp->naxes)[0];
   long DNY = (dst_stamp->naxes)[1];

   /* Initialize convolution kernel: */
   #define KXSIZE 2*halfx + 1
   #define KYSIZE 2*halfy + 1
   free_ezImg(kern_vals);                    /* just in case */
   init_ezImg(kern_vals);                    /* really blank */
   if ( blank_image_dimen(kern_vals, KXSIZE, KYSIZE) ) {
      fprintf(stderr, "%s: failed to allocate kernel image!\n", FUNCNAME);
      return 1; /* failure */
   }
 //n_pix_params += kern_vals->NumPix;   /* must fit for kernel pixels */
   #define NKERNPIX kern_vals->NumPix

   /* NULL background pointer disables fitting of background term: */
   long extra_params = 0;
   if ( background != NULL ) {
      fprintf(stderr, "%s: background fitting enabled!\n", FUNCNAME);
      extra_params += 1;
   } else {
      fprintf(stderr, "%s: background fitting disabled!\n", FUNCNAME);
   }

   /* Summarize fitted parameters: */
   const long total_params = NKERNPIX + extra_params;
   #define NFITPARS total_params
   fprintf(stderr, "%s:\n"
         "Kernel pixels:   %ld\n"
         "Extra params:    %ld\n"
         "Total params:    %ld\n"
         , FUNCNAME, NKERNPIX, extra_params, total_params);

   /* Kernel size is set by dimensions of kern_vals ezImg: */
   //const int halfx = (kern_vals->naxes[0] - 1) / 2;
   //const int halfy = (kern_vals->naxes[1] - 1) / 2;
   //int halfx = 7;    // half-size of kernel in X
   //int halfy = 7;    // half-size of kernel in Y

   /* If kernel includes a background adjustment term: */
 //#define NFITPARS (2*halfx + 1)*(2*halfy + 1) + 1
 //#define NFITPARS kern_vals->NumPix + 1
 //#define NKERNPIX KXSIZE*KYSIZE

   /* If matched backgrounds are assumed for the stamps: */
 //#define NFITPARS (2*halfx + 1)*(2*halfy + 1)
 //#define NFITPARS KXSIZE * KYSIZE
 //#define NFITPARS kern_vals->NumPix

   fprintf(stderr, "Beginning matrix/array allocation ... ");
   /* Parameters: */
   double *params = malloc(NFITPARS * sizeof(*params));
   for ( i = 0; i < NFITPARS; i++ ) { params[i] = 0.0; }

   /* Create intermediate vectors and matrices: */
   double XTy[NFITPARS]; // = { 0.0 };
   for ( i = 0; i < NFITPARS; i++ ) { XTy[i] = 0.0; }

   double **mXTX  = malloc(NFITPARS * sizeof(*mXTX));
   double **mXTXi = malloc(NFITPARS * sizeof(*mXTXi));
   double   *XTX  = malloc(NFITPARS * NFITPARS * sizeof(*XTX));
   double   *XTXi = malloc(NFITPARS * NFITPARS * sizeof(*XTXi));
   if ( !mXTX || !mXTXi || !XTX || !XTXi ) {
      fprintf(stderr, "%s: memory allocation failure!\n", FUNCNAME);
      if ( mXTX  != NULL ) free(mXTX );
      if ( mXTXi != NULL ) free(mXTXi);
      if (  XTX  != NULL ) free( XTX );
      if (  XTXi != NULL ) free( XTXi);
      return 1; /* failure */
   }
   for ( i = 0; i < NFITPARS; i++ ) {
      mXTX[i]  = XTX  + i*NFITPARS;
      mXTXi[i] = XTXi + i*NFITPARS;
   }

   /* Initialize: */
   for ( i = 0; i < NFITPARS*NFITPARS; i++ ) { XTX[i] = XTXi[i] = 0.0; }

   fprintf(stderr, "done.\n");

   /* ---------------------------------------------------------------------- */
   /* ---------------------------------------------------------------------- */
   /* ---------------------------------------------------------------------- */
   /* ---------------------------------------------------------------------- */

 ///* Pre-compute 1-D index offsets: */
 ////long *r_indices = malloc((NFITPARS - 1) * sizeof(*r_indices));
 ////long *r_indices = malloc(kern_vals->NumPix * sizeof(*r_indices));
 //long *idx_offsets = malloc(NKERNPIX * sizeof(*idx_offsets));
 //int k = 0;
 //for ( int j = -halfy; j <= halfy; j++ ) {          /* kernel rows */
 //   for ( int i = -halfx; i <= halfx; i++ ) {       /* kernel cols */
 //      idx_offsets[k++] = j * NCOLS + i;
 //   }
 //}

   print_size(src_stamp, stderr);
   fprintf(stderr, "src_stamp[0][0]: %f\n", src_stamp->pix2D[0][0]);
   print_size(dst_stamp, stderr);
   fprintf(stderr, "dst_stamp[0][0]: %f\n", dst_stamp->pix2D[0][0]);
   fprintf(stderr, "Starting parallel portion ... \n");
#ifdef _OPENMP
   omp_set_num_threads(1);
#endif

#ifdef _OPENMP
#  pragma omp parallel private(i, j, X, Y)
#endif
   {  /* beginning of OpenMP parallel region */

      /* Scratch data (per-thread): */
      register double dstpixval, srcpixval;
      register int x_oob, y_oob;
      //register long x_eff, y_eff;
      //double dX, dY;
      int    isbad[NFITPARS]; // = { 0 };
      double slice[NFITPARS]; // = { 1.0 };
      double tmp_XTy[NFITPARS]; // = { 0.0 };
      double tmp_mXTX[NFITPARS][NFITPARS]; // = {{ 0.0 }};

      /* Initialize (these arrays are variable-length): */
      fprintf(stderr, "Initialize stuff before sums ... ");
      for ( i = 0; i < NFITPARS; i++ ) {
         slice[i] = 1.0;
         isbad[i] = 0;
         tmp_XTy[i] = 0.0;
         for ( j = 0; j < NFITPARS; j++ ) { tmp_mXTX[i][j] = 0.0; }
      }
      fprintf(stderr, "done.\n");

      /* Loop over dst_stamp area (ignores edge pix): */
      fprintf(stderr, "Beginning loop over dst_stamp ... ");
#  ifdef _OPENMP
#     pragma omp for schedule(static)
#  endif
      for ( Y = halfy; Y < DNY - halfy; Y++ ) {            /* dst_stamp rows */
    //for ( Y = 1; Y < DNY - 1; Y++ ) {                    /* dst_stamp rows */
         for ( X = halfx; X < DNX - halfx; X++ ) {         /* dst_stamp cols */
       //for ( X = 1; X < DNX - 1; X++ ) {                 /* dst_stamp cols */

            dstpixval = (dst_stamp->pix2D)[Y][X];

            /* Skip bad pixels from dst_stamp: */
            if ( isnan(dstpixval) || isinf(dstpixval) ) { continue; }

            /* Fill slice (2-D): */
          //int pp = 1;
            int pp = extra_params;
            for ( j = -halfy; j <= halfy; j++ ) {          /* kernel rows */
               y_oob = (Y + j < 0) || (SNY <= Y + j);
               for ( i = -halfx; i <= halfx; i++ ) {       /* kernel cols */
                  x_oob = (X + i < 0) || (SNX <= X + i);
                  if ( y_oob || x_oob ) {
                     isbad[pp] = 1;    /* out-of-bounds, avoid segfault */
                  } else {
                     srcpixval = src_stamp->pix2D[Y+j][X+i];
                     slice[pp] = srcpixval;
                     isbad[pp] = isnan(srcpixval) || isinf(srcpixval);
                  }
                  pp++;
               }
            }

          ///* Fill slice (1-D): */
          //long idx1D = Y * NCOLS + X;
          //int i, pp = extra_params;
          //for ( i = 0; i < NKERNPIX; i++ ) {
          //   srcpixval = src_stamp->pix1D[idx1D + idx_offsets[i]];
          //}

            /* Compute XTy (X^T * img): */
            for ( i = 0; i < NFITPARS; i++ ) {
               if ( !isbad[i] ) { tmp_XTy[i] += slice[i] * dstpixval; }
            }

            /* Compute XTX (X^T * X): */
            for ( j = 0; j < NFITPARS; j++ ) {
               if ( !isbad[j] ) {
                  for ( i = 0; i < NFITPARS; i++ ) {
                     if ( !isbad[i] ) { tmp_mXTX[j][i] += slice[j] * slice[i]; }
                  }
               }
            }

         }              /* close loop over rows */
      }                 /* close loop over rows */
      fprintf(stderr, "done.\n");

#ifdef _OPENMP
#  pragma omp critical
#endif
      {
         /* Accumulate results: */
         for ( j = 0; j < NFITPARS; j++ ) { XTy[j] += tmp_XTy[j]; }
         for ( j = 0; j < NFITPARS; j++ ) {
            for ( i = 0; i < NFITPARS; i++ ) {
               mXTX[j][i] += tmp_mXTX[j][i];
            }
         }
      }

   } /* end of OpenMP parallel region (implied barrier) */


   /* Invert matrix: */
   fprintf(stderr, "Inverting matrix ... ");
   MatrixInversion(mXTX, NFITPARS, mXTXi);
   fprintf(stderr, "done.\n");

   /* Compute fit parameters: */
   for ( j = 0; j < NFITPARS; j++ ) {
      for ( i = 0; i < NFITPARS; i++ ) {
         params[j] += mXTXi[j][i] * XTy[i];
      }
   }

   /* Put fitted kernel pixels into ezImg: */
   fprintf(stderr, "Storing kernel pixels ... ");
   for ( i = 0; i < kern_vals->NumPix; i++ ) {
      kern_vals->pix1D[i] = params[i + extra_params];
   }
   fprintf(stderr, "done.\n");

   /* Return results: */
 //*fitpar = params;
 //*nparam = NFITPARS;
   if ( background != NULL ) { *background = params[0]; }

   /* Free scratch memory: */
   free(XTX);   free(mXTX);
   free(XTXi);  free(mXTXi);

   return 0; /* success */

}
