import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.core.Point;

public class RetinalMatch {
        public static void main(String[] args) {
                System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
                
                //Get the image filenames to compare from the command line
                String imagePath1 = args[0];
                String imagePath2 = args[1];

                //Make MAT objects using these filenames
                Mat image1 = Imgcodecs.imread(imagePath1);
                Mat image2 = Imgcodecs.imread(imagePath2);

                //Clean up these images so that they can be compared
                Mat cleanImage1 = cleanImage(image1);
                Mat cleanImage2 = cleanImage(image2);
        
		//Show these images
                Imgcodecs.imwrite("image1_output.jpg", cleanImage1);
                Imgcodecs.imwrite("image2_output.jpg", cleanImage2);

                List<Mat> bgrPlanes1 = new ArrayList<>();
                List<Mat> bgrPlanes2 = new ArrayList<>();
                Core.split(cleanImage1, bgrPlanes1);
                Core.split(cleanImage2, bgrPlanes2);

                //Create the histograms
                int histSize = 256;
                float[] range = {0, 256}; //the upper boundary is exclusive
                MatOfFloat histRange = new MatOfFloat(range);
                boolean accumulate = false;

                Mat histogramImage1 = new Mat();
                Mat histogramImage2 = new Mat();
                Imgproc.calcHist(bgrPlanes1, new MatOfInt(0), new Mat(), histogramImage1, new MatOfInt(histSize), histRange, accumulate);
                Imgproc.calcHist(bgrPlanes2, new MatOfInt(0), new Mat(), histogramImage2, new MatOfInt(histSize), histRange, accumulate);

                int histWidth = 512;
                int histHeight = 400;
                int binWidth = (int) Math.round((double) histWidth / histSize);
                Mat histImage1 = new Mat( histHeight, histWidth, CvType.CV_8U, new Scalar(0));
                Mat histImage2 = new Mat( histHeight, histWidth, CvType.CV_8U, new Scalar(0));

                //Normalize the histograms
                Core.normalize(histogramImage1, histogramImage1, 0, histImage1.rows(), Core.NORM_MINMAX);
                Core.normalize(histogramImage2, histogramImage2, 0, histImage2.rows(), Core.NORM_MINMAX);

                //Show the histograms
                float[] histogramImage1Data = new float[(int) (histogramImage1.total() * histogramImage1.channels())];
                histogramImage1.get(0, 0, histogramImage1Data);

                float[] histogramImage2Data = new float[(int) (histogramImage2.total() * histogramImage2.channels())];
                histogramImage2.get(0, 0, histogramImage2Data);

                for( int i = 1; i < histSize; i++ ) {
                        Imgproc.line(histImage1, new Point(binWidth * (i - 1), histHeight - Math.round(histogramImage1Data[i - 1])),
                            new Point(binWidth * (i), histHeight - Math.round(histogramImage1Data[i])), new Scalar(255), 2);
                        
                        Imgproc.line(histImage2, new Point(binWidth * (i - 1), histHeight - Math.round(histogramImage2Data[i - 1])),
                            new Point(binWidth * (i), histHeight - Math.round(histogramImage2Data[i])), new Scalar(255), 2);
                }

                Imgcodecs.imwrite("histogram1.jpg", histImage1);
                Imgcodecs.imwrite("histogram2.jpg", histImage2);

                //Compare the difference of the normalized histograms

	}

        private static Mat cleanImage(Mat src)
        {
                Mat dst=new Mat();
                Mat gray = new Mat();
                Mat edges = new Mat();

                Imgproc.cvtColor(src, src, Imgproc.COLOR_RGB2GRAY);
                //Imgproc.cvtColor(dst,dst,Imgproc.COLOR_RGB2GRAY);

                //Increasing the contrast of the image
                src.convertTo(src, -1, 1, 10);

                //Adding smoothing
                Imgproc.medianBlur(src, edges, 9);
                Imgproc.GaussianBlur(edges, edges, new Size(11, 11), 0);

                //Imgproc.GaussianBlur(gray, edges, new Size(11, 11), 0);
        
                //Detecting the edges
                //Imgproc.Canny(edges, edges, 4,12);
                Imgproc.Laplacian(edges, dst, CvType.CV_16S, 5, 0.6, 9, Core.BORDER_DEFAULT);

                Mat image = new Mat();
        
                Core.convertScaleAbs(dst, image);

                //Copying the detected edges to the destination matrix
                //src.copyTo(dst, edges);  
                Mat image2 = new Mat(src.rows(), src.cols(), src.type());

                Core.addWeighted(image, 1.2, image2, -0.5, 0, image2);

                //Setting to greyscale.
                Imgproc.threshold(image2, image2, 20, 255, Imgproc.THRESH_BINARY_INV);

                Imgproc.medianBlur(image2, image2, 9);

                return image2;
        }
}
