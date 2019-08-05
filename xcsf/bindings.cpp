#include <string>

extern "C" int xcsf_square(int);

extern "C" {   
#include <stdbool.h>
#include "cl.h"
#include "cl_set.h"
}

struct XCS
{
	XCSF xcs;
	XCS(std::string infname, int max_trials) {
		xcs.pop_num = max_trials;
	}
	int get_pop_num() {
		return xcs.pop_num;
	}
};

namespace {
  std::string greet() { return "hello, world"; }
}

#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(xcsf)
{
	class_<XCS>("XCS", init<std::string, int>())
		.def("get_pop_num", &XCS::get_pop_num);

    // Add regular functions to the module.
    def("greet", greet);
    def("square", xcsf_square);

}
