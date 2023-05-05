#include "bl_config.h"
#include "bl_dgemm_kernel.h"

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based micorkernel
//
void bl_dgemm_ukr( int    k,
		   int    m,
                   int    n,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < n; ++j )
        { 
            for ( i = 0; i < m; ++i )
            { 
	      c( i, j, ldc ) += a( l, i, m) * b( l, j, n );   
            }
        }
    }
}

void dgemm_16x4x4 (int lda, int ldb, int n, const double* restrict A, const double* restrict B, const double* restrict C, unsigned long long ldc, aux_t  *data)
{
    register svfloat64_t ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x, c8x, c9x, c10x, c11x, c12x, c13x, c14x, c15x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    c0x = svld1_f64(npred, C);
    c1x = svld1_f64(npred, C+ldc);
    c2x = svld1_f64(npred, C+(ldc*2));
    c3x = svld1_f64(npred, C+(ldc*3));
    c4x = svld1_f64(npred, C+(ldc*4));
    c5x = svld1_f64(npred, C+(ldc*5));
    c6x = svld1_f64(npred, C+(ldc*6));
    c7x = svld1_f64(npred, C+(ldc*7));
    c8x = svld1_f64(npred, C+(ldc*8));
    c9x = svld1_f64(npred, C+(ldc*9));
    c10x = svld1_f64(npred, C+(ldc*10));
    c11x = svld1_f64(npred, C+(ldc*11));
    c12x = svld1_f64(npred, C+(ldc*12));
    c13x = svld1_f64(npred, C+(ldc*13));
    c14x = svld1_f64(npred, C+(ldc*14));
    c15x = svld1_f64(npred, C+(ldc*15));
    register float64_t aval;
    for (int kk=0; kk<lda; kk++){
        aval = *(A + kk* DGEMM_MR);
        ax0 =svdup_f64(aval);
        aval = *(A + 1 + kk* DGEMM_MR);
        ax1 =svdup_f64(aval);
        aval = *(A + 2 + kk* DGEMM_MR);
        ax2 =svdup_f64(aval);
        aval = *(A + 3 + kk* DGEMM_MR);
        ax3 =svdup_f64(aval);
        aval = *(A + 4 + kk* DGEMM_MR);
        ax4 =svdup_f64(aval);
        aval = *(A + 5 + kk* DGEMM_MR);
        ax5 =svdup_f64(aval);
        aval = *(A + 6 + kk* DGEMM_MR);
        ax6 =svdup_f64(aval);
        aval = *(A + 7 + kk* DGEMM_MR);
        ax7 =svdup_f64(aval);
        aval = *(A + 8 + kk* DGEMM_MR);
        ax8 =svdup_f64(aval);
        aval = *(A + 9 + kk* DGEMM_MR);
        ax9 =svdup_f64(aval);
        aval = *(A + 10 + kk* DGEMM_MR);
        ax10 =svdup_f64(aval);
        aval = *(A + 11 + kk* DGEMM_MR);
        ax11 =svdup_f64(aval);
        aval = *(A + 12 + kk* DGEMM_MR);
        ax12 =svdup_f64(aval);
        aval = *(A + 13 + kk* DGEMM_MR);
        ax13 =svdup_f64(aval);
        aval = *(A + 14 + kk* DGEMM_MR);
        ax14 =svdup_f64(aval);
        aval = *(A + 15 + kk* DGEMM_MR);
        ax15 =svdup_f64(aval);

        bx = svld1_f64(svptrue_b64(), B + kk*ldb);
        c0x =svmla_f64_m(npred, c0x, bx, ax0);
        c1x =svmla_f64_m(npred, c1x, bx, ax1);
        c2x =svmla_f64_m(npred, c2x, bx, ax2);
        c3x =svmla_f64_m(npred, c3x, bx, ax3);
        c4x =svmla_f64_m(npred, c4x, bx, ax4);
        c5x =svmla_f64_m(npred, c5x, bx, ax5);
        c6x =svmla_f64_m(npred, c6x, bx, ax6);
        c7x =svmla_f64_m(npred, c7x, bx, ax7);
        c8x =svmla_f64_m(npred, c8x, bx, ax8);
        c9x =svmla_f64_m(npred, c9x, bx, ax9);
        c10x =svmla_f64_m(npred, c10x, bx, ax10);
        c11x =svmla_f64_m(npred, c11x, bx, ax11);
        c12x =svmla_f64_m(npred, c12x, bx, ax12);
        c13x =svmla_f64_m(npred, c13x, bx, ax13);
        c14x =svmla_f64_m(npred, c14x, bx, ax14);
        c15x =svmla_f64_m(npred, c15x, bx, ax15);

    }
    svst1_f64(npred, C, c0x);
    svst1_f64(npred, C+ldc, c1x);
    svst1_f64(npred, C+(ldc*2), c2x);
    svst1_f64(npred, C+(ldc*3), c3x);
    svst1_f64(npred, C+(ldc*4), c4x);
    svst1_f64(npred, C+(ldc*5), c5x);
    svst1_f64(npred, C+(ldc*6), c6x);
    svst1_f64(npred, C+(ldc*7), c7x);
    svst1_f64(npred, C+(ldc*8), c8x);
    svst1_f64(npred, C+(ldc*9), c9x);
    svst1_f64(npred, C+(ldc*10), c10x);
    svst1_f64(npred, C+(ldc*11), c11x);
    svst1_f64(npred, C+(ldc*12), c12x);
    svst1_f64(npred, C+(ldc*13), c13x);
    svst1_f64(npred, C+(ldc*14), c14x);
    svst1_f64(npred, C+(ldc*15), c15x);

}
void dgemm_14x4x4 (int lda, int ldb, int n, const double* restrict A, const double* restrict B, const double* restrict C, unsigned long long ldc, aux_t  *data)
{
    register svfloat64_t ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x, c8x, c9x, c10x, c11x, c12x, c13x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    c0x = svld1_f64(npred, C);
    c1x = svld1_f64(npred, C+ldc);
    c2x = svld1_f64(npred, C+(ldc*2));
    c3x = svld1_f64(npred, C+(ldc*3));
    c4x = svld1_f64(npred, C+(ldc*4));
    c5x = svld1_f64(npred, C+(ldc*5));
    c6x = svld1_f64(npred, C+(ldc*6));
    c7x = svld1_f64(npred, C+(ldc*7));
    c8x = svld1_f64(npred, C+(ldc*8));
    c9x = svld1_f64(npred, C+(ldc*9));
    c10x = svld1_f64(npred, C+(ldc*10));
    c11x = svld1_f64(npred, C+(ldc*11));
    c12x = svld1_f64(npred, C+(ldc*12));
    c13x = svld1_f64(npred, C+(ldc*13));
    register float64_t aval;
    for (int kk=0; kk<lda; kk++){
        aval = *(A + kk* DGEMM_MR);
        ax0 =svdup_f64(aval);
        aval = *(A + 1 + kk* DGEMM_MR);
        ax1 =svdup_f64(aval);
        aval = *(A + 2 + kk* DGEMM_MR);
        ax2 =svdup_f64(aval);
        aval = *(A + 3 + kk* DGEMM_MR);
        ax3 =svdup_f64(aval);
        aval = *(A + 4 + kk* DGEMM_MR);
        ax4 =svdup_f64(aval);
        aval = *(A + 5 + kk* DGEMM_MR);
        ax5 =svdup_f64(aval);
        aval = *(A + 6 + kk* DGEMM_MR);
        ax6 =svdup_f64(aval);
        aval = *(A + 7 + kk* DGEMM_MR);
        ax7 =svdup_f64(aval);
        aval = *(A + 8 + kk* DGEMM_MR);
        ax8 =svdup_f64(aval);
        aval = *(A + 9 + kk* DGEMM_MR);
        ax9 =svdup_f64(aval);
        aval = *(A + 10 + kk* DGEMM_MR);
        ax10 =svdup_f64(aval);
        aval = *(A + 11 + kk* DGEMM_MR);
        ax11 =svdup_f64(aval);
        aval = *(A + 12 + kk* DGEMM_MR);
        ax12 =svdup_f64(aval);
        aval = *(A + 13 + kk* DGEMM_MR);
        ax13 =svdup_f64(aval);

        bx = svld1_f64(svptrue_b64(), B + kk*ldb);
        c0x =svmla_f64_m(npred, c0x, bx, ax0);
        c1x =svmla_f64_m(npred, c1x, bx, ax1);
        c2x =svmla_f64_m(npred, c2x, bx, ax2);
        c3x =svmla_f64_m(npred, c3x, bx, ax3);
        c4x =svmla_f64_m(npred, c4x, bx, ax4);
        c5x =svmla_f64_m(npred, c5x, bx, ax5);
        c6x =svmla_f64_m(npred, c6x, bx, ax6);
        c7x =svmla_f64_m(npred, c7x, bx, ax7);
        c8x =svmla_f64_m(npred, c8x, bx, ax8);
        c9x =svmla_f64_m(npred, c9x, bx, ax9);
        c10x =svmla_f64_m(npred, c10x, bx, ax10);
        c11x =svmla_f64_m(npred, c11x, bx, ax11);
        c12x =svmla_f64_m(npred, c12x, bx, ax12);
        c13x =svmla_f64_m(npred, c13x, bx, ax13);

    }
    svst1_f64(npred, C, c0x);
    svst1_f64(npred, C+ldc, c1x);
    svst1_f64(npred, C+(ldc*2), c2x);
    svst1_f64(npred, C+(ldc*3), c3x);
    svst1_f64(npred, C+(ldc*4), c4x);
    svst1_f64(npred, C+(ldc*5), c5x);
    svst1_f64(npred, C+(ldc*6), c6x);
    svst1_f64(npred, C+(ldc*7), c7x);
    svst1_f64(npred, C+(ldc*8), c8x);
    svst1_f64(npred, C+(ldc*9), c9x);
    svst1_f64(npred, C+(ldc*10), c10x);
    svst1_f64(npred, C+(ldc*11), c11x);
    svst1_f64(npred, C+(ldc*12), c12x);
    svst1_f64(npred, C+(ldc*13), c13x);

}
void dgemm_12x4x4 (int lda, int ldb, int n, const double* restrict A, const double* restrict B, const double* restrict C, unsigned long long ldc, aux_t  *data)
{
    register svfloat64_t ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x, c8x, c9x, c10x, c11x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    c0x = svld1_f64(npred, C);
    c1x = svld1_f64(npred, C+ldc);
    c2x = svld1_f64(npred, C+(ldc*2));
    c3x = svld1_f64(npred, C+(ldc*3));
    c4x = svld1_f64(npred, C+(ldc*4));
    c5x = svld1_f64(npred, C+(ldc*5));
    c6x = svld1_f64(npred, C+(ldc*6));
    c7x = svld1_f64(npred, C+(ldc*7));
    c8x = svld1_f64(npred, C+(ldc*8));
    c9x = svld1_f64(npred, C+(ldc*9));
    c10x = svld1_f64(npred, C+(ldc*10));
    c11x = svld1_f64(npred, C+(ldc*11));
    register float64_t aval;
    for (int kk=0; kk<lda; kk++){
        aval = *(A + kk* DGEMM_MR);
        ax0 =svdup_f64(aval);
        aval = *(A + 1 + kk* DGEMM_MR);
        ax1 =svdup_f64(aval);
        aval = *(A + 2 + kk* DGEMM_MR);
        ax2 =svdup_f64(aval);
        aval = *(A + 3 + kk* DGEMM_MR);
        ax3 =svdup_f64(aval);
        aval = *(A + 4 + kk* DGEMM_MR);
        ax4 =svdup_f64(aval);
        aval = *(A + 5 + kk* DGEMM_MR);
        ax5 =svdup_f64(aval);
        aval = *(A + 6 + kk* DGEMM_MR);
        ax6 =svdup_f64(aval);
        aval = *(A + 7 + kk* DGEMM_MR);
        ax7 =svdup_f64(aval);
        aval = *(A + 8 + kk* DGEMM_MR);
        ax8 =svdup_f64(aval);
        aval = *(A + 9 + kk* DGEMM_MR);
        ax9 =svdup_f64(aval);
        aval = *(A + 10 + kk* DGEMM_MR);
        ax10 =svdup_f64(aval);
        aval = *(A + 11 + kk* DGEMM_MR);
        ax11 =svdup_f64(aval);

        bx = svld1_f64(svptrue_b64(), B + kk*ldb);
        c0x =svmla_f64_m(npred, c0x, bx, ax0);
        c1x =svmla_f64_m(npred, c1x, bx, ax1);
        c2x =svmla_f64_m(npred, c2x, bx, ax2);
        c3x =svmla_f64_m(npred, c3x, bx, ax3);
        c4x =svmla_f64_m(npred, c4x, bx, ax4);
        c5x =svmla_f64_m(npred, c5x, bx, ax5);
        c6x =svmla_f64_m(npred, c6x, bx, ax6);
        c7x =svmla_f64_m(npred, c7x, bx, ax7);
        c8x =svmla_f64_m(npred, c8x, bx, ax8);
        c9x =svmla_f64_m(npred, c9x, bx, ax9);
        c10x =svmla_f64_m(npred, c10x, bx, ax10);
        c11x =svmla_f64_m(npred, c11x, bx, ax11);

    }
    svst1_f64(npred, C, c0x);
    svst1_f64(npred, C+ldc, c1x);
    svst1_f64(npred, C+(ldc*2), c2x);
    svst1_f64(npred, C+(ldc*3), c3x);
    svst1_f64(npred, C+(ldc*4), c4x);
    svst1_f64(npred, C+(ldc*5), c5x);
    svst1_f64(npred, C+(ldc*6), c6x);
    svst1_f64(npred, C+(ldc*7), c7x);
    svst1_f64(npred, C+(ldc*8), c8x);
    svst1_f64(npred, C+(ldc*9), c9x);
    svst1_f64(npred, C+(ldc*10), c10x);
    svst1_f64(npred, C+(ldc*11), c11x);
}
void dgemm_10x4x4 (int lda, int ldb, int n, const double* restrict A, const double* restrict B, const double* restrict C, unsigned long long ldc, aux_t  *data)
{
    register svfloat64_t ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x, c8x, c9x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    c0x = svld1_f64(npred, C);
    c1x = svld1_f64(npred, C+ldc);
    c2x = svld1_f64(npred, C+(ldc*2));
    c3x = svld1_f64(npred, C+(ldc*3));
    c4x = svld1_f64(npred, C+(ldc*4));
    c5x = svld1_f64(npred, C+(ldc*5));
    c6x = svld1_f64(npred, C+(ldc*6));
    c7x = svld1_f64(npred, C+(ldc*7));
    c8x = svld1_f64(npred, C+(ldc*8));
    c9x = svld1_f64(npred, C+(ldc*9));
    register float64_t aval;
    for (int kk=0; kk<lda; kk++){
        aval = *(A + kk* DGEMM_MR);
        ax0 =svdup_f64(aval);
        aval = *(A + 1 + kk* DGEMM_MR);
        ax1 =svdup_f64(aval);
        aval = *(A + 2 + kk* DGEMM_MR);
        ax2 =svdup_f64(aval);
        aval = *(A + 3 + kk* DGEMM_MR);
        ax3 =svdup_f64(aval);
        aval = *(A + 4 + kk* DGEMM_MR);
        ax4 =svdup_f64(aval);
        aval = *(A + 5 + kk* DGEMM_MR);
        ax5 =svdup_f64(aval);
        aval = *(A + 6 + kk* DGEMM_MR);
        ax6 =svdup_f64(aval);
        aval = *(A + 7 + kk* DGEMM_MR);
        ax7 =svdup_f64(aval);
        aval = *(A + 8 + kk* DGEMM_MR);
        ax8 =svdup_f64(aval);
        aval = *(A + 9 + kk* DGEMM_MR);
        ax9 =svdup_f64(aval);

        bx = svld1_f64(svptrue_b64(), B + kk*ldb);
        c0x =svmla_f64_m(npred, c0x, bx, ax0);
        c1x =svmla_f64_m(npred, c1x, bx, ax1);
        c2x =svmla_f64_m(npred, c2x, bx, ax2);
        c3x =svmla_f64_m(npred, c3x, bx, ax3);
        c4x =svmla_f64_m(npred, c4x, bx, ax4);
        c5x =svmla_f64_m(npred, c5x, bx, ax5);
        c6x =svmla_f64_m(npred, c6x, bx, ax6);
        c7x =svmla_f64_m(npred, c7x, bx, ax7);
        c8x =svmla_f64_m(npred, c8x, bx, ax8);
        c9x =svmla_f64_m(npred, c9x, bx, ax9);
    }
    svst1_f64(npred, C, c0x);
    svst1_f64(npred, C+ldc, c1x);
    svst1_f64(npred, C+(ldc*2), c2x);
    svst1_f64(npred, C+(ldc*3), c3x);
    svst1_f64(npred, C+(ldc*4), c4x);
    svst1_f64(npred, C+(ldc*5), c5x);
    svst1_f64(npred, C+(ldc*6), c6x);
    svst1_f64(npred, C+(ldc*7), c7x);
    svst1_f64(npred, C+(ldc*8), c8x);
    svst1_f64(npred, C+(ldc*9), c9x);

}
void dgemm_8x4x4 (int lda, int ldb, int n, const double* restrict A, const double* restrict B, const double* restrict C, unsigned long long ldc, aux_t  *data)
{
    register svfloat64_t ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    c0x = svld1_f64(npred, C);
    c1x = svld1_f64(npred, C+ldc);
    c2x = svld1_f64(npred, C+(ldc*2));
    c3x = svld1_f64(npred, C+(ldc*3));
    c4x = svld1_f64(npred, C+(ldc*4));
    c5x = svld1_f64(npred, C+(ldc*5));
    c6x = svld1_f64(npred, C+(ldc*6));
    c7x = svld1_f64(npred, C+(ldc*7));

    register float64_t aval;
    for (int kk=0; kk<lda; kk++){
        aval = *(A + kk* DGEMM_MR);
        ax0 =svdup_f64(aval);
        aval = *(A + 1 + kk* DGEMM_MR);
        ax1 =svdup_f64(aval);
        aval = *(A + 2 + kk* DGEMM_MR);
        ax2 =svdup_f64(aval);
        aval = *(A + 3 + kk* DGEMM_MR);
        ax3 =svdup_f64(aval);
        aval = *(A + 4 + kk* DGEMM_MR);
        ax4 =svdup_f64(aval);
        aval = *(A + 5 + kk* DGEMM_MR);
        ax5 =svdup_f64(aval);
        aval = *(A + 6 + kk* DGEMM_MR);
        ax6 =svdup_f64(aval);
        aval = *(A + 7 + kk* DGEMM_MR);
        ax7 =svdup_f64(aval);


        bx = svld1_f64(svptrue_b64(), B + kk*ldb);
        c0x =svmla_f64_m(npred, c0x, bx, ax0);
        c1x =svmla_f64_m(npred, c1x, bx, ax1);
        c2x =svmla_f64_m(npred, c2x, bx, ax2);
        c3x =svmla_f64_m(npred, c3x, bx, ax3);
        c4x =svmla_f64_m(npred, c4x, bx, ax4);
        c5x =svmla_f64_m(npred, c5x, bx, ax5);
        c6x =svmla_f64_m(npred, c6x, bx, ax6);
        c7x =svmla_f64_m(npred, c7x, bx, ax7);
    }
    svst1_f64(npred, C, c0x);
    svst1_f64(npred, C+ldc, c1x);
    svst1_f64(npred, C+(ldc*2), c2x);
    svst1_f64(npred, C+(ldc*3), c3x);
    svst1_f64(npred, C+(ldc*4), c4x);
    svst1_f64(npred, C+(ldc*5), c5x);
    svst1_f64(npred, C+(ldc*6), c6x);
    svst1_f64(npred, C+(ldc*7), c7x);
}
void dgemm_6x4x4 (int lda, int ldb, int n, const double* restrict A, const double* restrict B, const double* restrict C, unsigned long long ldc, aux_t  *data)
{
    register svfloat64_t ax0, ax1, ax2, ax3, ax4, ax5;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    c0x = svld1_f64(npred, C);
    c1x = svld1_f64(npred, C+ldc);
    c2x = svld1_f64(npred, C+(ldc*2));
    c3x = svld1_f64(npred, C+(ldc*3));
    c4x = svld1_f64(npred, C+(ldc*4));
    c5x = svld1_f64(npred, C+(ldc*5));

    register float64_t aval;
    for (int kk=0; kk<lda; kk++){
        aval = *(A + kk* DGEMM_MR);
        ax0 =svdup_f64(aval);
        aval = *(A + 1 + kk* DGEMM_MR);
        ax1 =svdup_f64(aval);
        aval = *(A + 2 + kk* DGEMM_MR);
        ax2 =svdup_f64(aval);
        aval = *(A + 3 + kk* DGEMM_MR);
        ax3 =svdup_f64(aval);
        aval = *(A + 4 + kk* DGEMM_MR);
        ax4 =svdup_f64(aval);
        aval = *(A + 5 + kk* DGEMM_MR);
        ax5 =svdup_f64(aval);

        bx = svld1_f64(svptrue_b64(), B + kk*ldb);
        c0x =svmla_f64_m(npred, c0x, bx, ax0);
        c1x =svmla_f64_m(npred, c1x, bx, ax1);
        c2x =svmla_f64_m(npred, c2x, bx, ax2);
        c3x =svmla_f64_m(npred, c3x, bx, ax3);
        c4x =svmla_f64_m(npred, c4x, bx, ax4);
        c5x =svmla_f64_m(npred, c5x, bx, ax5);
 
    }
    svst1_f64(npred, C, c0x);
    svst1_f64(npred, C+ldc, c1x);
    svst1_f64(npred, C+(ldc*2), c2x);
    svst1_f64(npred, C+(ldc*3), c3x);
    svst1_f64(npred, C+(ldc*4), c4x);
    svst1_f64(npred, C+(ldc*5), c5x);

}
void dgemm_4x4x4 (int lda, int ldb, int n, const double* restrict A, const double* restrict B, const double* restrict C, unsigned long long ldc, aux_t  *data)
{
    register svfloat64_t ax0, ax1, ax2, ax3;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    c0x = svld1_f64(npred, C);
    c1x = svld1_f64(npred, C+ldc);
    c2x = svld1_f64(npred, C+(ldc*2));
    c3x = svld1_f64(npred, C+(ldc*3));
    register float64_t aval;
    for (int kk=0; kk<lda; kk++){
        aval = *(A + kk* DGEMM_MR);
        ax0 =svdup_f64(aval);
        aval = *(A + 1 + kk* DGEMM_MR);
        ax1 =svdup_f64(aval);
        aval = *(A + 2 + kk* DGEMM_MR);
        ax2 =svdup_f64(aval);
        aval = *(A + 3 + kk* DGEMM_MR);
        ax3 =svdup_f64(aval);

        bx = svld1_f64(svptrue_b64(), B + kk*ldb);
        c0x =svmla_f64_m(npred, c0x, bx, ax0);
        c1x =svmla_f64_m(npred, c1x, bx, ax1);
        c2x =svmla_f64_m(npred, c2x, bx, ax2);
        c3x =svmla_f64_m(npred, c3x, bx, ax3);
    }
    svst1_f64(npred, C, c0x);
    svst1_f64(npred, C+ldc, c1x);
    svst1_f64(npred, C+(ldc*2), c2x);
    svst1_f64(npred, C+(ldc*3), c3x);
}
void dgemm_2x4x4 (int lda, int ldb, int n, const double* restrict A, const double* restrict B, const double* restrict C, unsigned long long ldc, aux_t  *data)
{
    register svfloat64_t ax0, ax1;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    c0x = svld1_f64(npred, C);
    c1x = svld1_f64(npred, C+ldc);
    register float64_t aval;
    for (int kk=0; kk<lda; kk++){
        aval = *(A + kk* DGEMM_MR);
        ax0 =svdup_f64(aval);
        aval = *(A + 1 + kk* DGEMM_MR);
        ax1 =svdup_f64(aval);

        bx = svld1_f64(svptrue_b64(), B + kk*ldb);
        c0x =svmla_f64_m(npred, c0x, bx, ax0);
        c1x =svmla_f64_m(npred, c1x, bx, ax1);
    }
    svst1_f64(npred, C, c0x);
    svst1_f64(npred, C+ldc, c1x);
}
