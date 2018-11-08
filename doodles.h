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

#include "imageIO.h"

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
                     );

