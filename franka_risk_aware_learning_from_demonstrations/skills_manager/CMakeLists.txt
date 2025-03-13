cmake_minimum_required(VERSION 3.0.2)
project(skills_manager)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_python_setup()
catkin_package()


install(PROGRAMS
  src/skills_manager/superfile.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY src/skills_manager/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__init__.py"
)