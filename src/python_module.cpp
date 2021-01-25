#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#define NUMPY_IMPORT_ARRAY_RETVAL NULL

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include "sift.simd.hpp"
#include <iostream>

#include <algorithm>

const int nOctaveLayers = 3;
const float sigma = 1.6;

std::string getImageType(int number)
{
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}

static cv::Mat createInitialImage( const cv::Mat& img, bool doubleImageSize, float sigma )
{
    CV_TRACE_FUNCTION();

    cv::Mat gray, gray_fpt;
    if( img.channels() == 3 || img.channels() == 4 )
    {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray_fpt, cv::DataType<cv::sift_wt>::type, cv::SIFT_FIXPT_SCALE, 0);
    }
    else
        img.convertTo(gray_fpt, cv::DataType<cv::sift_wt>::type, cv::SIFT_FIXPT_SCALE, 0);

    float sig_diff;

    if( doubleImageSize )
    {
        sig_diff = sqrtf( std::max(sigma * sigma - cv::SIFT_INIT_SIGMA * cv::SIFT_INIT_SIGMA * 4, 0.01f) );
        cv::Mat dbl;
#if DoG_TYPE_SHORT
        cv::resize(gray_fpt, dbl, cv::Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, cv::INTER_LINEAR_EXACT);
#else
        cv::resize(gray_fpt, dbl, cv::Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, cv::INTER_LINEAR);
#endif
        cv::Mat result;
        cv::GaussianBlur(dbl, result, cv::Size(), sig_diff, sig_diff);
        return result;
    }
    else
    {
        sig_diff = sqrtf( std::max(sigma * sigma - cv::SIFT_INIT_SIGMA * cv::SIFT_INIT_SIGMA, 0.01f) );
        cv::Mat result;
        cv::GaussianBlur(gray_fpt, result, cv::Size(), sig_diff, sig_diff);
        return result;
    }
}

namespace pbcvt {

  using namespace boost::python;

  PyObject *sift_desc(PyObject *impy, PyObject *keypoints, int nOctaves) {
    // IDK if this is clear but there is a lot of opencv code in here

    cv::Mat im, kpts;
    im = pbcvt::fromNDArrayToMat(impy);
    kpts = pbcvt::fromNDArrayToMat(keypoints);
    printf("%i, %i\n", kpts.rows, kpts.cols);
    // First, build a gauss_pyr

    // Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
    // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
    // Copyright (C) 2020, Intel Corporation, all rights reserved.
    cv::Mat base = createInitialImage(im, false, sigma);
    CV_TRACE_FUNCTION();
    std::vector<cv::Mat> gauss_pyr;

    std::vector<double> sig(nOctaveLayers + 3);
    gauss_pyr.resize(nOctaves*(nOctaveLayers + 3));

    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sig[0] = sigma;
    double k = std::pow( 2., 1. / nOctaveLayers );
    for( int i = 1; i < nOctaveLayers + 3; i++ )
    {
        double sig_prev = std::pow(k, (double)(i-1))*sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    for( int o = 0; o < nOctaves; o++ )
    {
        for( int i = 0; i < nOctaveLayers + 3; i++ )
        {
            cv::Mat& dst = gauss_pyr[o*(nOctaveLayers + 3) + i];
            if( o == 0  &&  i == 0 )
                dst = base;
            // base of new octave is halved image from end of previous octave
            else if( i == 0 )
            {
                const cv::Mat& src = gauss_pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
                cv::resize(src, dst, cv::Size(src.cols/2, src.rows/2),
                       0, 0, cv::INTER_NEAREST);
            }
            else
            {
                const cv::Mat& src = gauss_pyr[o*(nOctaveLayers + 3) + i-1];
                cv::GaussianBlur(src, dst, cv::Size(), sig[i], sig[i]);
            }
        }
    }

    // Compute orientation of keypoints
    for (int i=0; i<kpts.rows; i++) {
      // Get attributes of keypoint
      float x = kpts.at<float>(i, 0);
      float y = kpts.at<float>(i, 1);
      float size = kpts.at<float>(i, 2);
      int layer = kpts.at<float>(i, 3);
      int octave = kpts.at<float>(i, 4);
      float c = x / (1 << octave);
      float r = y / (1 << octave);
      // printf("%f, %f, %f, %i, %i\n", x, y, size, layer, octave);

      // Init variables
      int n = cv::SIFT_ORI_HIST_BINS;
      float hist[n];
      float scl_octv = size*0.5f/(1 << octave);

      // std::cout << gauss_pyr[octave*(nOctaveLayers+3) + layer].depth() << std::endl;
      // cv::imshow("HI", gauss_pyr[octave*(nOctaveLayers+3) + layer]);
      // cv::waitKey(0);
      float omax = cv::opt_CV_CPU_DISPATCH_MODE::calcOrientationHist(
          gauss_pyr[octave*(nOctaveLayers+3) + layer],
          // cv::Point(c, r),
          cv::Point(20, 20),
          cvRound(cv::SIFT_ORI_RADIUS * scl_octv),
          cv::SIFT_ORI_SIG_FCTR * scl_octv,
          hist, n);

      float mag_thr = (float)(omax * cv::SIFT_ORI_PEAK_RATIO);
      for( int j = 0; j < n; j++ )
      {
          int l = j > 0 ? j - 1 : n - 1;
          int r2 = j < n-1 ? j + 1 : 0;

          if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
          {
              float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
              bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
              float angle = 360.f - (float)((360.f/n) * bin);
              if(std::abs(angle - 360.f) < FLT_EPSILON)
                  angle = 0.f;

              kpts.at<float>(i, 5) = angle;
              printf("%f\n", angle);
          }
      }
    }

    PyObject *ret = pbcvt::fromMatToNDArray(kpts);
    return ret;
  }

#if (PY_VERSION_HEX >= 0x03000000)

  static void *init_ar() {
#else
    static void init_ar(){
#endif
      Py_Initialize();

      import_array();
      return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (pbcvt) {
      //using namespace XM;
      init_ar();

      //initialize converters
      to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
      matFromNDArrayBoostConverter();

      //expose module-level functions
      def("sift_desc", sift_desc);
    }

  } //end namespace pbcvt
