
#include "simpleIO.h"

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

   /* Initialize convolution kernel: */
   #define KXSIZE 2*halfx + 1
   #define KYSIZE 2*halfy + 1
   free_ezImg(kern_vals);                    /* just in case */
   init_ezImg(kern_vals);                    /* really blank */
   if ( blank_image_dimen(kern_vals, KXSIZE, KYSIZE) ) {
      fprintf(stderr, "%s: failed to allocate kernel image!\n", FUNCNAME);
      return 1; /* failure */
   }
 //n_pix_params += kern_values->NumPix;   /* must fit for kernel pixels */
   #define NKERNPIX kern_values->NumPix

   /* NULL background pointer disables fitting of background term: */
   long extra_params = 0;
   if ( background != NULL ) {
      fprintf(stderr, "%s: background fitting enabled!\n", FUNCNAME);
      extra_params += 1;
   } else {
      fprintf(stderr, "%s: background fitting disabled!\n", FUNCNAME);
   }

   /* Summarize fitted parameters: */
   long total_parameters = NKERNPIX + extra_params;
   #define NFITPARS total_parameters
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

   /* Parameters: */
   double *params = malloc(NFITPARS * sizeof(*params));
   for ( int i = 0; i < NFITPARS; i++ ) { params[i] = 0.0; }

   /* Create intermediate vectors and matrices: */
   double XTy[NFITPARS] = { 0.0 };
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
   for ( int i = 0; i < NFITPARS; i++ ) {
      mXTX[i]  = XTX  + i*NFITPARS;
      mXTXi[i] = XTXi + i*NFITPARS;
   }

   /* Initialize: */
   for ( int i = 0; i < NFITPARS*NFITPARS; i++ ) { XTX[i] = XTXi[i] = 0.0; }

   /* ---------------------------------------------------------------------- */
   /* ---------------------------------------------------------------------- */
   /* ---------------------------------------------------------------------- */
   /* ---------------------------------------------------------------------- */

   /* Pre-compute 1-D index offsets: */
   //long *r_indices = malloc((NFITPARS - 1) * sizeof(*r_indices));
   //long *r_indices = malloc(kern_values->NumPix * sizeof(*r_indices));
   long *idx_offsets = malloc(NKERNPIX * sizeof(*idx_offsets));
   k = 0;
   for ( int j = -halfy; j <= halfy; j++ ) {          /* kernel rows */
      for ( int i = -halfx; i <= halfx; i++ ) {       /* kernel cols */
         idx_offsets[k++] = j * NCOLS + i;
      }
   }

#ifdef _OPENMP
#  pragma omp parallel private(i, j, X, Y)
#endif
   {  /* beginning of OpenMP parallel region */

      /* Scratch data (per-thread): */
      register double dstpixval, srcpixval;
      double dX, dY;
      double slice[NFITPARS] = { 1.0 };
      double tmp_XTy[NFITPARS] = { 0.0 };
      double tmp_mXTX[NFITPARS][NFITPARS] = {{ 0.0 }};
      int    isbad[NFITPARS] = { 0 };

   /* Loop over stamp area (ignores edge pix): */
#  ifdef _OPENMP
#     pragma omp for schedule(static)
#  endif
   for ( Y = 1; Y < NROWS - 1; Y++ ) {              /* loop over image rows */
      for ( X = 1; X < NCOLS - 1; X++ ) {           /* loop over image cols */

         dstpixval = (dst_stamp->pix2D)[Y][X];

         /* Skip bad pixels from dst_stamp: */
         if ( isnan(dstpixval) || isinf(dstpixval) ) { continue; }

         /* Fill slice (2-D): */
       //int pp = 1;
         int i, j, pp = extra_params;
         for ( j = -halfy; j <= halfy; j++ ) {          /* kernel rows */
            for ( i = -halfx; i <= halfx; i++ ) {       /* kernel cols */
               srcpixval = src_stamp->pix2D[Y+j][X+i];
               slice[pp] = srcpixval;
               smask[pp] = isnan(srcpixval) || isinf(srcpixval);
               pp++;
            }
         }

       ///* Fill slice (1-D): */
       //long idx1D = Y * NCOLS + X;
       //int i, pp = extra_params;
       //for ( i = 0; i < NKERNPIX; i++ ) {
       //   srcpixval = src_stamp->pix1D[idx1D + idx_offsets[i]];
       //}

         /* Add slice contents into intermediate sums, skip bad values: */
         //if ( !isnan(dstpixval) && !isinf(dstpixval) ) {
         //NOTE: this should really be "if pixval_okay AND sliceval_okay ...
         //
 
         /* Compute XTy (X^T * img): */
         for ( i = 0; i < NPARS; i++ ) {
            if ( !smask[i] ) {
               tmp_XTy[i] += slice[i] * dstpixval;
            }
         }

         /* Compute XTX (X^T * X): */
         for ( j = 0; j < NPARS; j++ ) {
            if ( !smask[j] ) {
               for ( i = 0; i < NPARS; i++ ) {
                  if ( !smask[i] ) {
                     tmp_mXTX[j][i] += slice[j] * slice[i];
                  }
               }
            }
         }

         //}         /* end of non-NaN block */

      }
   }

#ifdef _OPENMP
#  pragma omp critical
#endif
      {
         /* Accumulate results: */
         for ( j = 0; j < NPARS; j++ ) { XTy[j] += tmp_XTy[j]; }
         for ( j = 0; j < NPARS; j++ ) {
            for ( i = 0; i < NPARS; i++ ) {
               mXTX[j][i] += tmp_mXTX[j][i];
            }
         }
      }
   } /* end of OpenMP parallel region (implied barrier) */

   /* Invert matrix: */
   MatrixInversion(mXTX, NPARS, mXTXi);

   /* Compute fit parameters: */
   for ( j = 0; j < NPARS; j++ ) {
      for ( i = 0; i < NPARS; i++ ) {
         params[j] += mXTXi[j][i] * XTy[i];
      }
   }

   /* Return results: */
   *fitpar = params;
   *nparam = NPARS;

   /* Free scratch memory: */
   free(XTX);   free(mXTX);
   free(XTXi);  free(mXTXi);

   return 0; /* success */

}
