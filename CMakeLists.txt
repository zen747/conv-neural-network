cmake_minimum_required (VERSION 2.8.4)

project (conv-neural-network)

# common include directories
find_package(Boost)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
else()
    message("Boost not found! please define BOOST_ROOT")
endif()

set (LIBS dl rt)

add_executable (fizzbuzz fizzbuzz.cpp conv-neural-network.cpp )
target_link_libraries (fizzbuzz ${LIBS})
install (TARGETS fizzbuzz DESTINATION bin)

add_executable (conv-net conv-net.cpp conv-neural-network.cpp conv-neural-network.h)
target_link_libraries (conv-net ${LIBS} SDL2 SDL2_image)
install (TARGETS conv-net DESTINATION bin)

