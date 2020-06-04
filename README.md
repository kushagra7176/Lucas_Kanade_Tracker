	#############################################################################################
	#######        ENPM 673 - Perception for Autonomous Robots                           ########
	#######        Authors  @Kushagra Agrawal                                            ########
	#######        Project 4 : Lucas Kanade Tracker                                      ########
	#############################################################################################


	#########################################################################
	#                    #***************************#                      #
	#                    #*** Directory Structure ***#			#
	#                    #***************************#			#
	#     									#
	#									#
	# 	"kushagra7176.zip" 						#
	# (Extract this compressed file to any desired folder) 			#
	#	Folder Name after extraction: kushagra7176			#
	#	|-Code								#
	#	|	|-LK_Tracker.py					        #
	#	|	|-LK_Tracker_Weighted.py			       	#
	#                                                                       #
	#	|-Report.pdf							#
	#	|-README.md							#
	#									#
	#########################################################################

***************************** Special Instructions ****************************** 
-- Install all package dependencies like OpenCv(cv2) before running the code.
-- Update pip and all the packages to the latest versions.
-- Make sure you have the following files in your directory:
	1. LK_Tracker.py
	2. LK_Tracker_Weighted.py	
	3. DragonBaby    (dataset folder)
	4. Bolt2         (dataset folder)
	5. Car4          (dataset folder)
	6. Bolt2_ROI_coordinates.txt
	7. DragonBaby_ROI_coordinates.txt
	8. Car4_ROI_coordinates.txt


Instructions to run the file:

################## PyCharm IDE preferred #############

-> Extract the "kushagra7176.zip" file to any desired folder or location.

-> 1. Using Command Prompt:

->-> Navigate to the "Code" folder inside "kushagra7176" folder, right click and select open terminal or command prompt here. (Skip Step 2)
	*** use 'python3' instead of 'python' in the terminal if using Linux based OS ***
	Execute the files by using the following commands in command prompt or terminal:
	--python .\LK_Tracker.py 
	--python .\LK_Tracker_Weighted.py

-> 2. Using PyCharm IDE

	Open all the files in the IDE and click Run and select the "<filename.py>" to execute.

######### NOTE ###########
-- Output for car is in the video file OUTPUT_CAR_video.avi.
-- Output for bolt is in the video file OUTPUT_BOLT_video.avi.
-- Output for DragonBaby is in the video file OUTPUT_BABY_video.avi.
-- Output for car for the weighted algorithm is in the video file OUTPUT_CAR_video.avi.
