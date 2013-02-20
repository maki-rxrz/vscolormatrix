/*
**                 ColorMatrix v2.5 for Avisynth 2.5.x
**
**   ColorMatrix 2.0 is based on the original ColorMatrix filter by Wilbert 
**   Dijkhof.  It adds the ability to convert between any of: Rec.709, FCC, 
**   Rec.601, and SMPTE 240M. It also makes pre and post clipping optional,
**   adds range expansion/contraction, and more...
**
**   Copyright (C) 2006-2009 Kevin Stone
**
**   ColorMatrix 1.x is Copyright (C) Wilbert Dijkhof
**
**   This program is free software; you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation; either version 2 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program; if not, write to the Free Software
**   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#endif
#include <xmmintrin.h>
#include <cfloat>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include "VapourSynth.h"
#include "vshelper.h"

#define PLANAR_Y 0
#define PLANAR_U 1
#define PLANAR_V 2

#define VERSION "2.5"
#define DATE "01/25/2009"

#define MAGIC_NUMBER 0xdeadbeef
#define COLORIMETRY 0x0000001C
#define COLORIMETRY_SHIFT 2
#define CTS(n) n == 1 ? "Rec.709" : \
    n == 4 ? "FCC" : \
    n == 5 ? "Rec.601" : \
    n == 6 ? "Rec.601" : \
    n == 7 ? "SMPTE 240M" : \
    n == -1 ? "no hint found" : \
    "unknown"
#define CTS2(n) n == 1 ? "Rec.709->FCC" : \
    n == 2 ? "Rec.709->Rec.601" : \
    n == 3 ? "Rec.709->SMPTE 240M" : \
    n == 4 ? "FCC->Rec.709" : \
    n == 6 ? "FCC->Rec.601" : \
    n == 7 ? "FCC->SMPTE 240M" : \
    n == 8 ? "Rec.601->Rec.709" : \
    n == 9 ? "Rec.601->FCC" : \
    n == 11 ? "Rec.601->SMPTE 240M" : \
    n == 12 ? "SMPTE 240M->Rec.709" : \
    n == 13 ? "SMPTE 240M->FCC" : \
    n == 14 ? "SMPTE 240M->Rec.601" : \
    "unknown"
#define ns(n) n < 0 ? int(n*65536.0-0.5+DBL_EPSILON) : int(n*65536.0+0.5)
#define CB(n) (std::max)((std::min)((n),255),0)
#define simd_scale(n) n >= 65536 ? (n+2)>>2 : n >= 32768 ? (n+1)>>1 : n;

static double yuv_coeffs_luma[4][3] =
{ 
    +0.7152, +0.0722, +0.2126, // Rec.709 (0)
    +0.5900, +0.1100, +0.3000, // FCC (1)
    +0.5870, +0.1140, +0.2990, // Rec.601 (ITU-R BT.470-2/SMPTE 170M) (2)
    +0.7010, +0.0870, +0.2120, // SMPTE 240M (3)
};

__declspec(align(16)) const int64_t Q32[2] = { 0x0020002000200020, 0x0020002000200020 };
__declspec(align(16)) const int64_t Q64[2] = { 0x0040004000400040, 0x0040004000400040 };
__declspec(align(16)) const int64_t Q128[2] = { 0x0080008000800080, 0x0080008000800080 };
__declspec(align(16)) const int64_t Q8224[2] = { 0x2020202020202020, 0x2020202020202020 };
__declspec(align(16)) const int64_t Q8 = 0x0000000000000008;

struct CFS {
    int c1, c2, c3, c4;
    int c5, c6, c7, c8;
    int n, modef;
    int64_t cpu;
    bool debug;
};

struct PS_INFO {
    int ylut[256], uvlut[256];
    const unsigned char *srcp, *srcpn;
    const unsigned char *srcpU, *srcpV;
    int src_pitch, src_pitchR, src_pitchUV;
    int height, width, widtha;
    unsigned char *dstp, *dstpn;
    unsigned char *dstpU, *dstpV;
    int dst_pitch, dst_pitchR, dst_pitchUV;
    CFS *cs;
    HANDLE nextJob, jobFinished;
    bool finished;
};

enum {
    CACHE_NOTHING=0,
    CACHE_RANGE=1,
    CACHE_ALL=2,
    CACHE_AUDIO=3,
    CACHE_AUDIO_NONE=4
};

enum {                    
    /* slowest CPU to support extension */
    CPUF_FORCE			= 0x01,     // N/A
    CPUF_FPU			= 0x02,     // 386/486DX
    CPUF_MMX			= 0x04,     // P55C, K6, PII
    CPUF_INTEGER_SSE    = 0x08,		// PIII, Athlon
    CPUF_SSE			= 0x10,		// PIII, Athlon XP/MP
    CPUF_SSE2			= 0x20,		// PIV, Hammer
    CPUF_3DNOW			= 0x40,     // K6-2
    CPUF_3DNOW_EXT		= 0x80,		// Athlon
    CPUF_X86_64         = 0xA0,     // Hammer (note: equiv. to 3DNow + SSE2, which only Hammer
                                    // will have anyway)
    CPUF_SSE3		    = 0x100,    // Some P4 & Athlon 64.
};

int num_processors();
unsigned VS_CC processFrame_YUY2(void *ps);
unsigned VS_CC processFrame_YV12(void *ps);
void (*find_YV12_SIMD(int modef, bool sse2))(void *ps);
void conv1_YV12_MMX(void *ps);
void conv2_YV12_MMX(void *ps);
void conv3_YV12_MMX(void *ps);
void conv4_YV12_MMX(void *ps);
void conv1_YV12_SSE2(void *ps);
void conv2_YV12_SSE2(void *ps);
void conv3_YV12_SSE2(void *ps);
void conv4_YV12_SSE2(void *ps);

class ColorMatrix
{
private:
    int yuv_convert[16][3][3];
    const char *mode, *d2v;
    unsigned char *d2vArray;
    bool hints, interlaced, debug;
    bool inputFR, outputFR;
    int source, dest, modei, clamp;
    int opt, threads, thrdmthd;
    VSNodeRef *child;
    VSNodeRef *hintClip;
    VSVideoInfo vi;
    CFS css;
    unsigned *tids;
    HANDLE *thds;
    PS_INFO **pssInfo;
    int max_luma;
    int min_luma;
    int max_chroma;
    int min_chroma;

    void getHint(const unsigned char *srcp, int &color);
    void checkMode(const char *md, const VSAPI *vsapi);
    int findMode(int color);
    int parseD2V(const char *d2v);
    void inverse3x3(double im[3][3], double m[3][3]);
    void solve_coefficients(double cm[3][3], double rgb[3][3], double yuv[3][3],
        double yiscale, double uviscale, double yoscale, double uvoscale);
    void calc_coefficients(const VSAPI *vsapi);
    static int get_num_processors();

public:
    ColorMatrix(VSNodeRef *_child, const char* _mode, int _source, int _dest, 
        int _clamp, bool _interlaced, bool _inputFR, bool _outputFR, bool _hints, 
        const char* _d2v, bool _debug, int _threads, int _thrdmthd, int _opt, 
        const VSAPI *vsapi, VSCore *core);
    ~ColorMatrix();
    static const VSFrameRef *VS_CC ColorMatrixGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi);
    const VSFrameRef *getFrame(int n, int activationReason, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi);
    static void VS_CC ColorMatrixInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi);
    static void VS_CC ColorMatrixFree(void *instanceData, VSCore *core, const VSAPI *vsapi);
};