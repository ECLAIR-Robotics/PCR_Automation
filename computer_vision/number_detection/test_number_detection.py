import unittest
import os
import cv2
from number_detection import read_img
import time
import argparse

filePath = './test/test-cropped/'
txtPath = './test/testResults/'

class TestNumberDetection(unittest.TestCase):
    def test_function_multiple_times(self):
        # go through each file in the cropped directory
        # parse through the name and extract the expected value from the file name
        # anything in ()'s is diregarded
        fails = 0
        passes = 0
        results = []

        writeFile = self.writeFile

        for imgFile in os.listdir(filePath):
            #only get the first three digits with hyphen
            imgExpected = imgFile.split('(')[0].split('.jpg')[0].split('.png')[0]

            # print("File Name: " + imgFile)
            # print("Expected Value: " + imgExpected)

            #not completely sure about this syntax
            with self.subTest(imgFile=imgFile, imgExpected=imgExpected):
                img = cv2.imread(filePath+imgFile)
                result = read_img(img)

                try:
                    self.assertEqual(result,imgExpected)
                    passes += 1
                    results.append(f"{imgFile:<15} PASSED! RESULT: {result:<8}")
                except AssertionError as e:
                    fails += 1
                    results.append(f"{imgFile:<15} FAILED! RESULT: {result:<8} EXPECTED: {imgExpected:<5}")
            
        # Sort results: FAILED first, then PASSED
        results.sort(key=lambda x: "PASSED" in x)
        
        pass_rate = passes / (passes + fails) if (passes + fails) > 0 else 0
        summary = f"PASSES: {passes} \t FAILS: {fails} \t RATE: {pass_rate:.2f}\n"
        
        currentTime = time.strftime('%m-%d_%H-%M-%S')

        with open(writeFile, "w") as f:
            f.write(f"Test started at: {currentTime}\n")
            f.write("\nTest Results:\n")
            for result in results:
                f.write(result + "\n")
            f.write("\nSummary:\n")
            f.write(summary)
            #END OF METHOD
        
       

    def test_function_once(self):

        imgFile = self.imgFile
        imgExpected = imgFile.split('(')[0].split('.jpg')[0].split('.png')[0]

        img = cv2.imread(filePath+imgFile)

        print("File Name: " + imgFile)
        print("Expected Value: " + imgExpected)

        #not completely sure about this syntax
        with self.subTest(img=img,imgFile=imgFile, imgExpected=imgExpected):
            # img = cv2.imread(filePath+imgFile)
            result = read_img(img)
            self.assertEqual(result,imgExpected,
                            imgFile +" FAILED! RESULT: " + result + " EXPECTED: " + imgExpected)
        
        print('\n')


'''
MULTIPLE TESTS: set 'runMultiple' to True and run "python test-number-detection.py"
SINGLE TEST: set 'runMultiple' to False and run "python test-number-detection.py [FILENAME]"
'''

def main(runMultiple=True):
    os.makedirs(txtPath, exist_ok=True)

    # Get the current time and format it
    currentTime = time.strftime('%m-%d_%H-%M-%S')
    print(f"Current time: {currentTime}")

    parser = argparse.ArgumentParser(description="Run single test")
    parser.add_argument("imgFile", type=str, nargs="?", default="15.5.jpg", help="The image file we wish to crop")
    arg = parser.parse_args()

    # Open the file with the current time in the filename

    # file_path = f'{txtPath}{arg.imgFile}_result_{currentTime}.txt'
    file_path = f'{txtPath}FULL_result_{currentTime}.txt' if runMultiple else f'{txtPath}{arg.imgFile}_result_{currentTime}.txt'
    # Create a test suite and add the test case
    suite = unittest.TestSuite()
    
    # NOTE: To 
    if (runMultiple):
        test_case = TestNumberDetection('test_function_multiple_times')
        test_case.writeFile = file_path
    else:
        test_case = TestNumberDetection('test_function_once')
        test_case.imgFile = arg.imgFile
        print(f"PATH: {filePath+arg.imgFile}")
    
    
    suite.addTest(test_case)
    
    # Create a test runner that writes to the file
    runner = unittest.TextTestRunner(verbosity=0)
    
    # Run the tests
    runner.run(suite)


if __name__ == '__main__':
    main()
