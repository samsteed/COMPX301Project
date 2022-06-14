import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;

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
