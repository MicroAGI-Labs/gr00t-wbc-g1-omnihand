find_library(AUDIO_THREAD_ZMQ_LIBRARY zmq)
if(NOT AUDIO_THREAD_ZMQ_LIBRARY)
  message(FATAL_ERROR "libzmq is required by audio_thread")
endif()

add_library(audio_thread audio_thread.cpp)
target_include_directories(audio_thread PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(audio_thread PRIVATE unitree_sdk2 ${AUDIO_THREAD_ZMQ_LIBRARY})
