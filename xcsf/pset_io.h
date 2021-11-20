#include "../lib/pybind11/include/pybind11/numpy.h"
#include "../lib/pybind11/include/pybind11/pybind11.h"
#include <fstream>
#include <string>
#include <vector>

namespace py = pybind11;

extern "C" {
#include "act_integer.h"
#include "cl.h"
#include "clset.h"
#include "cond_rectangle.h"
#include "pred_nlms.h"
#include "utils.h"
}

bool
json_string_compare(const char *json_string, const std::string &compare_string)
{
    return (static_cast<std::string>(json_string).compare(compare_string) == 0);
}

/// @todo: Do this here or add it in the python file or in the json file?
void
set_params(struct XCSF *xcs)
{
    cond_param_set_type(xcs, COND_TYPE_HYPERRECTANGLE);
    action_param_set_type(xcs, ACT_TYPE_INTEGER);
    pred_param_set_type(xcs, PRED_TYPE_NLMS_LINEAR);

    /// @todo: Is this necessary? The print of array size for center and spread
    /// (which differ from each other) depend on this. If the value is too low,
    /// we got slicing. If the value is too high it gets filled with random
    /// values (doesn't break though)
    // xcs->x_dim = 10;
}

void
pset_write_to_file(struct XCSF *xcs, const std::string &filename)
{
    set_params(xcs);
    std::ofstream pset_json(filename);

    pset_json << clset_json(xcs, &xcs->pset, true, true, true);
    pset_json.close();
}

template <typename T>
void
fill_pset_param(T &pset_parameter, const cJSON *json_element,
                const std::string &string_param)
{
    T json_element_value;
    if (typeid(T) == typeid(double)) {
        json_element_value = json_element->valuedouble;
    } else if (typeid(T) == typeid(int)) {
        json_element_value = json_element->valueint;
    } else if (typeid(T) == typeid(bool)) {
        json_element_value = static_cast<bool>(json_element->valueint);
    }
    if (json_string_compare(json_element->string, string_param)) {
        pset_parameter = json_element_value;
    }
}

void
fill_pset_array(double *&pset_array, const cJSON *json_element,
                const std::string &string_param)
{
    if (json_string_compare(json_element->string, string_param)) {

        // necessary to allocate enough memory and copy them to pset_array
        std::vector<double> temp_vector;

        cJSON *json_array;
        cJSON_ArrayForEach(json_array, json_element)
        {
            temp_vector.push_back(json_array->valuedouble);
        }

        pset_array = (double *) malloc(sizeof(double) * temp_vector.size());

        for (std::size_t i = 0; i < temp_vector.size(); ++i) {
            pset_array[i] = temp_vector[i];
        }
    }
}

void
fill_pset_condition(Cl *&cl, const cJSON *json_element)
{
    if (json_string_compare(json_element->string, "condition")) {
        CondRectangle *cl_cond =
            static_cast<CondRectangle *>(malloc(sizeof(CondRectangle)));

        cJSON *condition;
        cJSON_ArrayForEach(condition, json_element)
        {
            fill_pset_array(cl_cond->center, condition, "center");
            fill_pset_array(cl_cond->spread, condition, "spread");
            fill_pset_array(cl_cond->mu, condition, "mutation");
        }

        cl->cond = cl_cond;
    }
}

void
fill_pset_action(Cl *&cl, const cJSON *json_element)
{
    if (json_string_compare(json_element->string, "action")) {
        ActInteger *cl_act =
            static_cast<ActInteger *>(malloc(sizeof(ActInteger)));

        cJSON *action;
        cJSON_ArrayForEach(action, json_element)
        {
            fill_pset_param(cl_act->action, action, "action");
            fill_pset_array(cl_act->mu, action, "mutation");
        }
        cl->act = cl_act;
    }
}

std::string
get_pset_string(const std::string &filename)
{
    std::string reader;
    std::string pset_string;

    std::ifstream pset_file(filename);

    while (getline(pset_file, reader)) {
        pset_string += reader;
    }

    pset_file.close();

    return pset_string;
}

int
get_classifier_amount(cJSON *classifiers)
{
    int classifier_counter = 0;

    cJSON *classifier;
    cJSON_ArrayForEach(classifier, classifiers)
    {
        classifier_counter++;
    }

    return classifier_counter;
}

void
fill_classifier_list(struct XCSF *xcs, cJSON *classifiers)
{
    int classifier_counter = 0;
    Clist *list = static_cast<Clist *>(
        malloc(sizeof(Clist) * get_classifier_amount(classifiers)));

    cJSON *classifier;
    cJSON_ArrayForEach(classifier, classifiers)
    {
        Cl *cl = static_cast<Cl *>(malloc(sizeof(Cl)));
        cl_rand(xcs, cl);

        cJSON *elem;
        cJSON_ArrayForEach(elem, classifier)
        {
            fill_pset_param(cl->err, elem, "error");
            fill_pset_param(cl->fit, elem, "fitness");
            fill_pset_param(cl->num, elem, "numerosity");
            fill_pset_param(cl->exp, elem, "experience");
            fill_pset_param(cl->size, elem, "set_size");
            fill_pset_param(cl->time, elem, "time");
            fill_pset_param(cl->m, elem, "current_match");
            fill_pset_param(cl->action, elem, "current_action");
            fill_pset_param(cl->age, elem, "samples_seen");
            fill_pset_param(cl->mtotal, elem, "samples_matched");
            fill_pset_array(cl->prediction, elem, "current_prediction");
            fill_pset_condition(cl, elem);
            fill_pset_action(cl, elem);
        }

        list[classifier_counter++].cl = cl;
    }
    xcs->pset.list = list;
}

void
pset_load_from_file(struct XCSF *xcs, const std::string &filename)
{
    set_params(xcs);

    const std::string pset_string = get_pset_string(filename);

    cJSON *root = cJSON_Parse(pset_string.c_str());
    cJSON *classifiers = cJSON_GetObjectItemCaseSensitive(root, "classifiers");

    clset_init(&xcs->pset);
    fill_classifier_list(xcs, classifiers);
}
