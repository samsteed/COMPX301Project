import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core.MinMaxLocResult;
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
import org.opencv.core.Rect;

public class RetinalMatch {
        public static void main(String[] args) {

                try{
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

                //Masking image-----------------------------------------------------
                //Drawing a Circle
                Imgproc.circle (cleanImage1, new Point(cleanImage1.cols()/2-cleanImage1.cols()/50, cleanImage1.rows()/2+cleanImage1.rows()/50), 540, new Scalar(255, 255, 255), 54);
                Imgproc.circle (cleanImage2, new Point(cleanImage2.cols()/2-cleanImage2.cols()/50, cleanImage2.rows()/2+cleanImage2.rows()/50), 540, new Scalar(255, 255, 255), 54);

                //Drawing Rectangle
                Imgproc.rectangle(cleanImage1, new Point(50, 60), new Point(cleanImage1.cols()-40, cleanImage1.rows()-40), new Scalar(255,255,255), 41);
                Imgproc.rectangle(cleanImage2, new Point(50, 60), new Point(cleanImage2.cols()-40, cleanImage2.rows()-40), new Scalar(255,255,255), 42);

                //Splitting images up/creating templates ------------------------------------
                Rect rect = new Rect(cleanImage1.cols()/8, cleanImage1.rows()/16, cleanImage1.cols() - cleanImage1.cols()/4-cleanImage1.cols()/30, cleanImage1.rows() - cleanImage1.rows()/12);
                Rect rect2 = new Rect(0, 0, rect.width/2, rect.height/2);
                Rect rect3 = new Rect(rect.width/2, 0, rect.width/2, rect.height/2);
                Rect rect4 = new Rect(0, rect.height/2, rect.width/2, rect.height/2);
                Rect rect5 = new Rect(rect.width/2, rect.height/2, rect.width/2, rect.height/2);


                Mat croppedImage1 = new Mat(cleanImage1, rect);
                Mat top_left1 = new Mat(croppedImage1, rect2);
                Mat top_right1 = new Mat(croppedImage1, rect3);
                Mat bottom_left1 = new Mat(croppedImage1, rect4);
                Mat bottom_right1 = new Mat(croppedImage1, rect5);
        
        
		//Showing cropped parts of image 1
                
                Imgcodecs.imwrite("cropped_image_1.jpg", croppedImage1);
                Imgcodecs.imwrite("top_left_1.jpg", top_left1);
                Imgcodecs.imwrite("top_right_1.jpg", top_right1);
                Imgcodecs.imwrite("bottom_left_1.jpg", bottom_left1);
                Imgcodecs.imwrite("bottom_right_1.jpg", bottom_right1);

                //Splitting image for image 2
                rect = new Rect(cleanImage2.cols()/8, cleanImage2.rows()/16, cleanImage2.cols() - cleanImage2.cols()/4-cleanImage2.cols()/30, cleanImage2.rows() - cleanImage2.rows()/12);
                rect2 = new Rect(0, 0, rect.width/2, rect.height/2);
                rect3 = new Rect(rect.width/2, 0, rect.width/2, rect.height/2);
                rect4 = new Rect(0, rect.height/2, rect.width/2, rect.height/2);
                rect5 = new Rect(rect.width/2, rect.height/2, rect.width/2, rect.height/2);

                Mat croppedImage2 = new Mat(cleanImage2, rect);
                Mat top_left2 = new Mat(croppedImage2, rect2);
                Mat top_right2 = new Mat(croppedImage2, rect3);
                Mat bottom_left2 = new Mat(croppedImage2, rect4);
                Mat bottom_right2 = new Mat(croppedImage2, rect5);
        
        
		//Showing cropped parts of image 2
                Imgcodecs.imwrite("cropped_image_2.jpg", croppedImage2);
                Imgcodecs.imwrite("top_left_2.jpg", top_left2);
                Imgcodecs.imwrite("top_right_2.jpg", top_right2);
                Imgcodecs.imwrite("bottom_left_2.jpg", bottom_left2);
                Imgcodecs.imwrite("bottom_right_2.jpg", bottom_right2);

                //Show these images
                Imgcodecs.imwrite("image1_output.jpg", cleanImage1);
                Imgcodecs.imwrite("image2_output.jpg", cleanImage2);

                

                //Trialing Template matching----------------------------------------
                Mat source = cleanImage1;
                Mat template = bottom_left2;

                double[] matchedValues = new double[4];

                Mat[] templates = new Mat[4];
                templates[0] = bottom_left2;
                templates[1] = bottom_right2;
                templates[2] = top_right2;
                templates[3] = top_left2;

                Mat outputImage = new Mat();
                int machMethod = Imgproc.TM_CCOEFF_NORMED;

                for(int i = 0; i<templates.length; i++){
                        template = templates[i];
                        source = cleanImage1;
                        Imgproc.matchTemplate(source, template, outputImage, machMethod);

                        MinMaxLocResult mmr = Core.minMaxLoc(outputImage);
                        Point matchLoc = mmr.maxLoc;


                        //Checking if match is above threshhold
                        System.out.println("Max Val: "+ mmr.maxVal);

                        if(mmr.maxVal > 0.4) {
                                //Adding success to array
                                matchedValues[i]=mmr.maxVal;
                                //Draw Rectangle on result image
                                Imgproc.rectangle(source, matchLoc, new Point(matchLoc.x + template.cols(),
                                matchLoc.y + template.rows()), new Scalar(0, 0, 0));
                        }

                        Imgcodecs.imwrite("template_match"+i+".jpg", source);
                }

                double bestMatch = 0;
                for(int i = 0; i<matchedValues.length; i++){
                        if(matchedValues[i]>bestMatch) bestMatch = matchedValues[i];
                }

                if(bestMatch!=0)System.out.println(bestMatch+"\n1");
                else System.out.println("No match: "+ 0);
                

                

                //------------------------------------------------------------------

                //Create the histograms
                List<Mat> bgrPlanes1 = new ArrayList<>();
                List<Mat> bgrPlanes2 = new ArrayList<>();
                Core.split(cleanImage1, bgrPlanes1);
                Core.split(cleanImage2, bgrPlanes2);

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

                histImage1.convertTo(histImage1, CvType.CV_32F);
                histImage2.convertTo(histImage2, CvType.CV_32F);

                //Compare the difference of the normalized histograms
                double correlation = Imgproc.compareHist(histImage1, histImage2, Imgproc.HISTCMP_CORREL);

                // if (correlation >= 0.99)
                // {
                //         System.out.println("1");
                // }
                // else
                // {
                //         System.out.println("0");
                // }

                }
                catch (Exception e){
                        System.out.println("Please enter a valid path and filename");
                }
                
	}

        private static Mat cleanImage(Mat src)
        {
                Mat dst=new Mat();
                Mat gray = new Mat();
                Mat edges = new Mat();

                Imgproc.cvtColor(src, src, Imgproc.COLOR_RGB2GRAY);

                //Increasing the contrast of the image
                src.convertTo(src, -1, 1, 10);

                //Adding smoothing
                Imgproc.medianBlur(src, edges, 9);
                Imgproc.GaussianBlur(edges, edges, new Size(11, 11), 0);
        
                //Detecting the edges
                //Imgproc.Canny(edges, edges, 4,12);
                Imgproc.Laplacian(edges, dst, CvType.CV_16S, 5, 0.6, 9, Core.BORDER_DEFAULT);

                Mat image = new Mat();
        
                Core.convertScaleAbs(dst, image);

                //Copying the detected edges to the destination matrix
                //src.copyTo(dst, edges);  
                Mat image2 = new Mat(src.rows(), src.cols(), src.type());

                Core.addWeighted(image, 1.2, image2, -0.5, 0, image2);

                //Setting to binary scale.
                Imgproc.threshold(image2, image2, 20, 255, Imgproc.THRESH_BINARY_INV);

                Imgproc.medianBlur(image2, image2, 9);

                return image2;
        }
}

//Template matching code
//https://riptutorial.com/opencv/example/22915/template-matching-with-java

//Template matching
//https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

//https://oleksandrg.medium.com/how-to-divide-the-image-into-4-parts-using-opencv-c0afb5cab10c
