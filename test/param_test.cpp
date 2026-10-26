/*
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file param_test.cpp
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2026.
 * @brief Parameter setting tests.
 */

#include "../lib/doctest/doctest/doctest.h"
#include <string>
#include <unistd.h>

extern "C" {
#include "../xcsf/action.h"
#include "../xcsf/loss.h"
#include "../xcsf/param.h"
#include "../xcsf/utils.h"
#include "../xcsf/xcsf.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
}

/* RAII wrapper so every test case initialises and frees params consistently */
struct ParamFixture {
    struct XCSF xcsf;
    ParamFixture(int x_dim = 5, int y_dim = 2, int n_actions = 1)
    {
        param_init(&xcsf, x_dim, y_dim, n_actions);
    }
    ~ParamFixture()
    {
        param_free(&xcsf);
    }
    bool
    ok() const
    {
        return xcsf.status == STATUS_OK;
    }
    /* "first error wins", so the error must be cleared between invalid calls */
    void
    reset()
    {
        xcsf_clear_error(&xcsf);
    }
};

/* Helpers for the repetitive "valid sets / invalid errors and keeps value" */
#define CHECK_INT_SETTER(setter, field, good, bad)                             \
    do {                                                                       \
        f.reset();                                                             \
        setter(&f.xcsf, (good));                                               \
        CHECK(f.ok());                                                         \
        CHECK_EQ(f.xcsf.field, (good));                                        \
        setter(&f.xcsf, (bad));                                                \
        CHECK_FALSE(f.ok());                                                   \
        CHECK_EQ(f.xcsf.field, (good));                                        \
        f.reset();                                                             \
    } while (0)

#define CHECK_DOUBLE_SETTER(setter, field, good, bad)                          \
    do {                                                                       \
        f.reset();                                                             \
        setter(&f.xcsf, (good));                                               \
        CHECK(f.ok());                                                         \
        CHECK(f.xcsf.field == doctest::Approx(good));                          \
        setter(&f.xcsf, (bad));                                                \
        CHECK_FALSE(f.ok());                                                   \
        CHECK(f.xcsf.field == doctest::Approx(good));                          \
        f.reset();                                                             \
    } while (0)

TEST_CASE("PARAM: defaults after init")
{
    ParamFixture f(5, 2, 3);
    CHECK(f.ok());
    CHECK_EQ(f.xcsf.x_dim, 5);
    CHECK_EQ(f.xcsf.y_dim, 2);
    CHECK_EQ(f.xcsf.n_actions, 3);
    CHECK_EQ(f.xcsf.time, 0);
    CHECK_EQ(f.xcsf.OMP_NUM_THREADS, 8);
    CHECK_EQ(f.xcsf.MAX_TRIALS, 100000);
    CHECK_EQ(f.xcsf.PERF_TRIALS, 1000);
    CHECK_EQ(f.xcsf.POP_SIZE, 2000);
    CHECK_EQ(f.xcsf.LOSS_FUNC, LOSS_MAE);
    CHECK(f.xcsf.POP_INIT);
    CHECK(f.xcsf.STATEFUL);
    CHECK_FALSE(f.xcsf.COMPACTION);
    CHECK_FALSE(f.xcsf.SET_SUBSUMPTION);
    CHECK_EQ(f.xcsf.THETA_SUB, 100);
    CHECK_EQ(f.xcsf.THETA_DEL, 20);
    CHECK_EQ(f.xcsf.M_PROBATION, 10000);
    CHECK_EQ(f.xcsf.TELETRANSPORTATION, 50);
    CHECK(f.xcsf.HUBER_DELTA == doctest::Approx(1));
    CHECK(f.xcsf.E0 == doctest::Approx(0.01));
    CHECK(f.xcsf.ALPHA == doctest::Approx(0.1));
    CHECK(f.xcsf.NU == doctest::Approx(5));
    CHECK(f.xcsf.BETA == doctest::Approx(0.1));
    CHECK(f.xcsf.DELTA == doctest::Approx(0.1));
    CHECK(f.xcsf.GAMMA == doctest::Approx(0.95));
    CHECK(f.xcsf.P_EXPLORE == doctest::Approx(0.9));
    CHECK(f.xcsf.INIT_FITNESS == doctest::Approx(0.01));
    CHECK(f.xcsf.INIT_ERROR == doctest::Approx(0));
    CHECK_EQ(strlen(f.xcsf.population_file), 0);
}

TEST_CASE("PARAM: dimension setters")
{
    ParamFixture f;
    CHECK_INT_SETTER(param_set_x_dim, x_dim, 10, 0);
    CHECK_INT_SETTER(param_set_x_dim, x_dim, 10, -1);
    CHECK_INT_SETTER(param_set_y_dim, y_dim, 4, 0);
    CHECK_INT_SETTER(param_set_y_dim, y_dim, 4, -1);
    CHECK_INT_SETTER(param_set_n_actions, n_actions, 3, 0);
    CHECK_INT_SETTER(param_set_n_actions, n_actions, 3, -1);
    /* boundary: smallest valid value */
    param_set_x_dim(&f.xcsf, 1);
    param_set_y_dim(&f.xcsf, 1);
    param_set_n_actions(&f.xcsf, 1);
    CHECK(f.ok());
}

TEST_CASE("PARAM: omp_num_threads")
{
    ParamFixture f;
    f.reset();
    param_set_omp_num_threads(&f.xcsf, 1);
    CHECK(f.ok());
    CHECK_EQ(f.xcsf.OMP_NUM_THREADS, 1);
    param_set_omp_num_threads(&f.xcsf, 1000);
    CHECK(f.ok());
    CHECK_EQ(f.xcsf.OMP_NUM_THREADS, 1000);
    param_set_omp_num_threads(&f.xcsf, 4);
    CHECK_EQ(f.xcsf.OMP_NUM_THREADS, 4);

    param_set_omp_num_threads(&f.xcsf, 0);
    CHECK_FALSE(f.ok());
    CHECK_EQ(f.xcsf.OMP_NUM_THREADS, 4);
    f.reset();
    param_set_omp_num_threads(&f.xcsf, 1001);
    CHECK_FALSE(f.ok());
    CHECK_EQ(f.xcsf.OMP_NUM_THREADS, 4);
    f.reset();
    param_set_omp_num_threads(&f.xcsf, -5);
    CHECK_FALSE(f.ok());
    CHECK_EQ(f.xcsf.OMP_NUM_THREADS, 4);
}

TEST_CASE("PARAM: random_state")
{
    ParamFixture f;
    param_set_random_state(&f.xcsf, 42);
    CHECK_EQ(f.xcsf.RANDOM_STATE, 42);
    CHECK(f.ok());

    /* same seed gives the same random sequence */
    param_set_random_state(&f.xcsf, 7);
    const double a = rand_uniform(0, 1);
    param_set_random_state(&f.xcsf, 7);
    const double b = rand_uniform(0, 1);
    CHECK(a == doctest::Approx(b));

    /* negative seeds are allowed and mean "seed from time" */
    param_set_random_state(&f.xcsf, -1);
    CHECK_EQ(f.xcsf.RANDOM_STATE, -1);
    CHECK(f.ok());
}

TEST_CASE("PARAM: population_file")
{
    ParamFixture f;
    param_set_population_file(&f.xcsf, "pop.bin");
    CHECK_EQ(strcmp(f.xcsf.population_file, "pop.bin"), 0);

    /* replacing frees the old buffer and stores the new one */
    param_set_population_file(&f.xcsf, "another_population_file.bin");
    CHECK_EQ(strcmp(f.xcsf.population_file, "another_population_file.bin"), 0);

    param_set_population_file(&f.xcsf, "");
    CHECK_EQ(strlen(f.xcsf.population_file), 0);

    /* overly long names are truncated and still null terminated */
    char longname[2048];
    memset(longname, 'a', sizeof(longname));
    longname[sizeof(longname) - 1] = '\0';
    param_set_population_file(&f.xcsf, longname);
    CHECK_LE(strlen(f.xcsf.population_file), 512);
}

TEST_CASE("PARAM: boolean setters")
{
    ParamFixture f;
    param_set_pop_init(&f.xcsf, false);
    CHECK_FALSE(f.xcsf.POP_INIT);
    param_set_pop_init(&f.xcsf, true);
    CHECK(f.xcsf.POP_INIT);

    param_set_stateful(&f.xcsf, false);
    CHECK_FALSE(f.xcsf.STATEFUL);
    param_set_stateful(&f.xcsf, true);
    CHECK(f.xcsf.STATEFUL);

    param_set_compaction(&f.xcsf, true);
    CHECK(f.xcsf.COMPACTION);
    param_set_compaction(&f.xcsf, false);
    CHECK_FALSE(f.xcsf.COMPACTION);

    param_set_set_subsumption(&f.xcsf, true);
    CHECK(f.xcsf.SET_SUBSUMPTION);
    param_set_set_subsumption(&f.xcsf, false);
    CHECK_FALSE(f.xcsf.SET_SUBSUMPTION);

    param_set_explore(&f.xcsf, true);
    CHECK(f.xcsf.explore);
    param_set_explore(&f.xcsf, false);
    CHECK_FALSE(f.xcsf.explore);

    CHECK(f.ok());
}

TEST_CASE("PARAM: positive integer setters")
{
    ParamFixture f;
    CHECK_INT_SETTER(param_set_max_trials, MAX_TRIALS, 500, 0);
    CHECK_INT_SETTER(param_set_max_trials, MAX_TRIALS, 500, -1);
    CHECK_INT_SETTER(param_set_perf_trials, PERF_TRIALS, 50, 0);
    CHECK_INT_SETTER(param_set_perf_trials, PERF_TRIALS, 50, -1);
    CHECK_INT_SETTER(param_set_pop_size, POP_SIZE, 100, 0);
    CHECK_INT_SETTER(param_set_pop_size, POP_SIZE, 100, -1);
    CHECK_INT_SETTER(param_set_teletransportation, TELETRANSPORTATION, 10, 0);
    CHECK_INT_SETTER(param_set_teletransportation, TELETRANSPORTATION, 10, -1);
}

TEST_CASE("PARAM: non-negative integer setters")
{
    ParamFixture f;
    CHECK_INT_SETTER(param_set_theta_del, THETA_DEL, 30, -1);
    CHECK_INT_SETTER(param_set_m_probation, M_PROBATION, 25, -1);
    CHECK_INT_SETTER(param_set_theta_sub, THETA_SUB, 15, -1);

    /* zero is valid for each */
    param_set_theta_del(&f.xcsf, 0);
    param_set_m_probation(&f.xcsf, 0);
    param_set_theta_sub(&f.xcsf, 0);
    CHECK(f.ok());
    CHECK_EQ(f.xcsf.THETA_DEL, 0);
    CHECK_EQ(f.xcsf.M_PROBATION, 0);
    CHECK_EQ(f.xcsf.THETA_SUB, 0);
}

TEST_CASE("PARAM: non-negative double setters")
{
    ParamFixture f;
    CHECK_DOUBLE_SETTER(param_set_huber_delta, HUBER_DELTA, 2.5, -0.1);
    CHECK_DOUBLE_SETTER(param_set_e0, E0, 0.05, -0.1);
    CHECK_DOUBLE_SETTER(param_set_init_error, INIT_ERROR, 0.3, -0.1);
    CHECK_DOUBLE_SETTER(param_set_init_fitness, INIT_FITNESS, 0.3, -0.1);
    CHECK_DOUBLE_SETTER(param_set_nu, NU, 3.0, -0.1);

    /* zero is valid, and values above 1 are valid for these */
    param_set_huber_delta(&f.xcsf, 0);
    param_set_e0(&f.xcsf, 0);
    param_set_init_error(&f.xcsf, 0);
    param_set_init_fitness(&f.xcsf, 0);
    param_set_nu(&f.xcsf, 100);
    CHECK(f.ok());
    CHECK(f.xcsf.NU == doctest::Approx(100));
}

TEST_CASE("PARAM: unit interval double setters")
{
    ParamFixture f;
    CHECK_DOUBLE_SETTER(param_set_gamma, GAMMA, 0.5, -0.01);
    CHECK_DOUBLE_SETTER(param_set_gamma, GAMMA, 0.5, 1.01);
    CHECK_DOUBLE_SETTER(param_set_p_explore, P_EXPLORE, 0.5, -0.01);
    CHECK_DOUBLE_SETTER(param_set_p_explore, P_EXPLORE, 0.5, 1.01);
    CHECK_DOUBLE_SETTER(param_set_alpha, ALPHA, 0.5, -0.01);
    CHECK_DOUBLE_SETTER(param_set_alpha, ALPHA, 0.5, 1.01);
    CHECK_DOUBLE_SETTER(param_set_beta, BETA, 0.5, -0.01);
    CHECK_DOUBLE_SETTER(param_set_beta, BETA, 0.5, 1.01);
    CHECK_DOUBLE_SETTER(param_set_delta, DELTA, 0.5, -0.01);
    CHECK_DOUBLE_SETTER(param_set_delta, DELTA, 0.5, 1.01);

    /* inclusive bounds */
    param_set_gamma(&f.xcsf, 0);
    param_set_gamma(&f.xcsf, 1);
    param_set_p_explore(&f.xcsf, 0);
    param_set_p_explore(&f.xcsf, 1);
    param_set_alpha(&f.xcsf, 0);
    param_set_alpha(&f.xcsf, 1);
    param_set_beta(&f.xcsf, 0);
    param_set_beta(&f.xcsf, 1);
    param_set_delta(&f.xcsf, 0);
    param_set_delta(&f.xcsf, 1);
    CHECK(f.ok());
}

TEST_CASE("PARAM: loss function")
{
    ParamFixture f;

    /* by enum */
    param_set_loss_func(&f.xcsf, LOSS_HUBER);
    CHECK(f.ok());
    CHECK_EQ(f.xcsf.LOSS_FUNC, LOSS_HUBER);
    param_set_loss_func(&f.xcsf, LOSS_MAE);
    CHECK_EQ(f.xcsf.LOSS_FUNC, LOSS_MAE);

    param_set_loss_func(&f.xcsf, -1);
    CHECK_FALSE(f.ok());
    CHECK_EQ(f.xcsf.LOSS_FUNC, LOSS_MAE);
    f.reset();
    param_set_loss_func(&f.xcsf, LOSS_NUM);
    CHECK_FALSE(f.ok());
    CHECK_EQ(f.xcsf.LOSS_FUNC, LOSS_MAE);
    f.reset();

    /* by string, using the library's own names so the test can't drift */
    param_set_loss_func_string(&f.xcsf, loss_type_as_string(LOSS_HUBER));
    CHECK(f.ok());
    CHECK_EQ(f.xcsf.LOSS_FUNC, LOSS_HUBER);
    param_set_loss_func_string(&f.xcsf, loss_type_as_string(LOSS_MAE));
    CHECK(f.ok());
    CHECK_EQ(f.xcsf.LOSS_FUNC, LOSS_MAE);

    param_set_loss_func_string(&f.xcsf, "not_a_loss_function");
    CHECK_FALSE(f.ok());
    f.reset();
}

TEST_CASE("PARAM: first error wins")
{
    ParamFixture f;
    param_set_pop_size(&f.xcsf, 0);
    CHECK_FALSE(f.ok());
    char first[sizeof(f.xcsf.err_msg)];
    strncpy(first, f.xcsf.err_msg, sizeof(first));
    first[sizeof(first) - 1] = '\0';

    param_set_max_trials(&f.xcsf, 0);
    CHECK_FALSE(f.ok());
    CHECK_EQ(strcmp(first, f.xcsf.err_msg), 0);

    f.reset();
    CHECK(f.ok());
    param_set_max_trials(&f.xcsf, 0);
    CHECK_FALSE(f.ok());
    CHECK_NE(strcmp(first, f.xcsf.err_msg), 0);
}

TEST_CASE("PARAM: json parse check")
{
    ParamFixture f;
    cJSON *good = cJSON_Parse("{\"x_dim\": 3}");
    param_json_parse_check(&f.xcsf, good);
    CHECK(f.ok());
    cJSON_Delete(good);

    cJSON *bad = cJSON_Parse("{\"x_dim\": ");
    CHECK(bad == NULL);
    param_json_parse_check(&f.xcsf, bad);
    CHECK_FALSE(f.ok());
}

TEST_CASE("PARAM: json import of individual parameters")
{
    ParamFixture f(5, 2, 2);
    const char *json_str = "{"
                           "\"x_dim\": 7,"
                           "\"y_dim\": 3,"
                           "\"n_actions\": 4,"
                           "\"omp_num_threads\": 2,"
                           "\"random_state\": 11,"
                           "\"population_file\": \"imported.bin\","
                           "\"pop_size\": 321,"
                           "\"max_trials\": 654,"
                           "\"pop_init\": false,"
                           "\"perf_trials\": 77,"
                           "\"loss_func\": \"huber\","
                           "\"huber_delta\": 2.5,"
                           "\"teletransportation\": 12,"
                           "\"gamma\": 0.8,"
                           "\"p_explore\": 0.7,"
                           "\"set_subsumption\": true,"
                           "\"theta_sub\": 33,"
                           "\"alpha\": 0.2,"
                           "\"beta\": 0.3,"
                           "\"delta\": 0.4,"
                           "\"nu\": 6,"
                           "\"theta_del\": 44,"
                           "\"init_fitness\": 0.05,"
                           "\"init_error\": 0.06,"
                           "\"e0\": 0.07,"
                           "\"m_probation\": 88,"
                           "\"stateful\": false,"
                           "\"compaction\": true"
                           "}";
    param_json_import(&f.xcsf, json_str);
    CHECK(f.ok());
    CHECK_EQ(f.xcsf.x_dim, 7);
    CHECK_EQ(f.xcsf.y_dim, 3);
    CHECK_EQ(f.xcsf.n_actions, 4);
    CHECK_EQ(f.xcsf.OMP_NUM_THREADS, 2);
    CHECK_EQ(f.xcsf.RANDOM_STATE, 11);
    CHECK_EQ(strcmp(f.xcsf.population_file, "imported.bin"), 0);
    CHECK_EQ(f.xcsf.POP_SIZE, 321);
    CHECK_EQ(f.xcsf.MAX_TRIALS, 654);
    CHECK_FALSE(f.xcsf.POP_INIT);
    CHECK_EQ(f.xcsf.PERF_TRIALS, 77);
    CHECK_EQ(f.xcsf.LOSS_FUNC, LOSS_HUBER);
    CHECK(f.xcsf.HUBER_DELTA == doctest::Approx(2.5));
    CHECK_EQ(f.xcsf.TELETRANSPORTATION, 12);
    CHECK(f.xcsf.GAMMA == doctest::Approx(0.8));
    CHECK(f.xcsf.P_EXPLORE == doctest::Approx(0.7));
    CHECK(f.xcsf.SET_SUBSUMPTION);
    CHECK_EQ(f.xcsf.THETA_SUB, 33);
    CHECK(f.xcsf.ALPHA == doctest::Approx(0.2));
    CHECK(f.xcsf.BETA == doctest::Approx(0.3));
    CHECK(f.xcsf.DELTA == doctest::Approx(0.4));
    CHECK(f.xcsf.NU == doctest::Approx(6));
    CHECK_EQ(f.xcsf.THETA_DEL, 44);
    CHECK(f.xcsf.INIT_FITNESS == doctest::Approx(0.05));
    CHECK(f.xcsf.INIT_ERROR == doctest::Approx(0.06));
    CHECK(f.xcsf.E0 == doctest::Approx(0.07));
    CHECK_EQ(f.xcsf.M_PROBATION, 88);
    CHECK_FALSE(f.xcsf.STATEFUL);
    CHECK(f.xcsf.COMPACTION);
}

TEST_CASE("PARAM: json import errors")
{
    /* unknown key */
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"not_a_param\": 1}");
        CHECK_FALSE(f.ok());
    }
    /* known key with the wrong value type */
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"x_dim\": \"abc\"}");
        CHECK_FALSE(f.ok());
    }
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"pop_init\": 1}");
        CHECK_FALSE(f.ok());
    }
    /* known key with an out of range value */
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"gamma\": 2.0}");
        CHECK_FALSE(f.ok());
    }
    /* invalid loss function name */
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"loss_func\": \"nonsense\"}");
        CHECK_FALSE(f.ok());
    }
    /* component sections: missing type, wrong type, invalid type */
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"condition\": {\"args\": {}}}");
        CHECK_FALSE(f.ok());
    }
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"condition\": {\"type\": 5}}");
        CHECK_FALSE(f.ok());
    }
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"condition\": {\"type\": \"bogus\"}}");
        CHECK_FALSE(f.ok());
    }
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"prediction\": {\"args\": {}}}");
        CHECK_FALSE(f.ok());
    }
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"prediction\": {\"type\": 5}}");
        CHECK_FALSE(f.ok());
    }
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"prediction\": {\"type\": \"bogus\"}}");
        CHECK_FALSE(f.ok());
    }
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"action\": {\"args\": {}}}");
        CHECK_FALSE(f.ok());
    }
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"action\": {\"type\": 5}}");
        CHECK_FALSE(f.ok());
    }
    {
        ParamFixture f;
        param_json_import(&f.xcsf, "{\"action\": {\"type\": \"bogus\"}}");
        CHECK_FALSE(f.ok());
    }
}

TEST_CASE("PARAM: json export and import round trip")
{
    ParamFixture a(6, 3, 1);
    param_set_random_state(&a.xcsf, 5);
    param_set_omp_num_threads(&a.xcsf, 3);
    param_set_population_file(&a.xcsf, "round_trip.bin");
    param_set_pop_init(&a.xcsf, false);
    param_set_max_trials(&a.xcsf, 1234);
    param_set_perf_trials(&a.xcsf, 56);
    param_set_pop_size(&a.xcsf, 789);
    param_set_loss_func(&a.xcsf, LOSS_HUBER);
    param_set_huber_delta(&a.xcsf, 1.5);
    param_set_set_subsumption(&a.xcsf, true);
    param_set_theta_sub(&a.xcsf, 17);
    param_set_e0(&a.xcsf, 0.02);
    param_set_alpha(&a.xcsf, 0.25);
    param_set_nu(&a.xcsf, 4);
    param_set_beta(&a.xcsf, 0.15);
    param_set_delta(&a.xcsf, 0.35);
    param_set_theta_del(&a.xcsf, 21);
    param_set_init_fitness(&a.xcsf, 0.03);
    param_set_init_error(&a.xcsf, 0.04);
    param_set_m_probation(&a.xcsf, 99);
    param_set_stateful(&a.xcsf, false);
    param_set_compaction(&a.xcsf, true);
    REQUIRE(a.ok());

    char *json_str = param_json_export(&a.xcsf);
    REQUIRE(json_str != NULL);

    ParamFixture b(1, 1, 1);
    param_json_import(&b.xcsf, json_str);
    CHECK(b.ok());
    CHECK_EQ(b.xcsf.x_dim, a.xcsf.x_dim);
    CHECK_EQ(b.xcsf.y_dim, a.xcsf.y_dim);
    CHECK_EQ(b.xcsf.n_actions, a.xcsf.n_actions);
    CHECK_EQ(b.xcsf.OMP_NUM_THREADS, a.xcsf.OMP_NUM_THREADS);
    CHECK_EQ(b.xcsf.RANDOM_STATE, a.xcsf.RANDOM_STATE);
    CHECK_EQ(strcmp(b.xcsf.population_file, a.xcsf.population_file), 0);
    CHECK_EQ(b.xcsf.POP_INIT, a.xcsf.POP_INIT);
    CHECK_EQ(b.xcsf.MAX_TRIALS, a.xcsf.MAX_TRIALS);
    CHECK_EQ(b.xcsf.PERF_TRIALS, a.xcsf.PERF_TRIALS);
    CHECK_EQ(b.xcsf.POP_SIZE, a.xcsf.POP_SIZE);
    CHECK_EQ(b.xcsf.LOSS_FUNC, a.xcsf.LOSS_FUNC);
    CHECK(b.xcsf.HUBER_DELTA == doctest::Approx(a.xcsf.HUBER_DELTA));
    CHECK_EQ(b.xcsf.SET_SUBSUMPTION, a.xcsf.SET_SUBSUMPTION);
    CHECK_EQ(b.xcsf.THETA_SUB, a.xcsf.THETA_SUB);
    CHECK(b.xcsf.E0 == doctest::Approx(a.xcsf.E0));
    CHECK(b.xcsf.ALPHA == doctest::Approx(a.xcsf.ALPHA));
    CHECK(b.xcsf.NU == doctest::Approx(a.xcsf.NU));
    CHECK(b.xcsf.BETA == doctest::Approx(a.xcsf.BETA));
    CHECK(b.xcsf.DELTA == doctest::Approx(a.xcsf.DELTA));
    CHECK_EQ(b.xcsf.THETA_DEL, a.xcsf.THETA_DEL);
    CHECK(b.xcsf.INIT_FITNESS == doctest::Approx(a.xcsf.INIT_FITNESS));
    CHECK(b.xcsf.INIT_ERROR == doctest::Approx(a.xcsf.INIT_ERROR));
    CHECK_EQ(b.xcsf.M_PROBATION, a.xcsf.M_PROBATION);
    CHECK_EQ(b.xcsf.STATEFUL, a.xcsf.STATEFUL);
    CHECK_EQ(b.xcsf.COMPACTION, a.xcsf.COMPACTION);

    /* re-exporting the imported params must give identical json */
    char *json_str2 = param_json_export(&b.xcsf);
    CHECK_EQ(strcmp(json_str, json_str2), 0);

    free(json_str);
    free(json_str2);
}

TEST_CASE("PARAM: json export is conditional on n_actions and loss")
{
    {
        ParamFixture f(3, 1, 1);
        param_set_loss_func(&f.xcsf, LOSS_MAE);
        char *s = param_json_export(&f.xcsf);
        cJSON *j = cJSON_Parse(s);
        CHECK((cJSON_GetObjectItem(j, "gamma") == NULL));
        CHECK((cJSON_GetObjectItem(j, "teletransportation") == NULL));
        CHECK((cJSON_GetObjectItem(j, "p_explore") == NULL));
        CHECK((cJSON_GetObjectItem(j, "huber_delta") == NULL));
        CHECK((cJSON_GetObjectItem(j, "version") != NULL));
        CHECK((cJSON_GetObjectItem(j, "condition") != NULL));
        CHECK((cJSON_GetObjectItem(j, "prediction") != NULL));
        cJSON_Delete(j);
        free(s);
    }
    {
        ParamFixture f(3, 1, 4);
        param_set_loss_func(&f.xcsf, LOSS_HUBER);
        char *s = param_json_export(&f.xcsf);
        cJSON *j = cJSON_Parse(s);
        CHECK((cJSON_GetObjectItem(j, "gamma") != NULL));
        CHECK((cJSON_GetObjectItem(j, "teletransportation") != NULL));
        CHECK((cJSON_GetObjectItem(j, "p_explore") != NULL));
        CHECK((cJSON_GetObjectItem(j, "huber_delta") != NULL));
        cJSON_Delete(j);
        free(s);
    }
}

TEST_CASE("PARAM: serialization")
{
    ParamFixture a(6, 3, 2);
    param_set_population_file(&a.xcsf, "saved.bin");
    param_set_pop_init(&a.xcsf, false);
    param_set_max_trials(&a.xcsf, 4321);
    param_set_perf_trials(&a.xcsf, 12);
    param_set_pop_size(&a.xcsf, 345);
    param_set_loss_func(&a.xcsf, LOSS_HUBER);
    param_set_huber_delta(&a.xcsf, 1.25);
    param_set_gamma(&a.xcsf, 0.6);
    param_set_teletransportation(&a.xcsf, 9);
    param_set_p_explore(&a.xcsf, 0.4);
    param_set_set_subsumption(&a.xcsf, true);
    param_set_theta_sub(&a.xcsf, 14);
    param_set_e0(&a.xcsf, 0.03);
    param_set_alpha(&a.xcsf, 0.12);
    param_set_nu(&a.xcsf, 2);
    param_set_beta(&a.xcsf, 0.22);
    param_set_delta(&a.xcsf, 0.32);
    param_set_theta_del(&a.xcsf, 8);
    param_set_init_fitness(&a.xcsf, 0.05);
    param_set_init_error(&a.xcsf, 0.07);
    param_set_m_probation(&a.xcsf, 123);
    param_set_stateful(&a.xcsf, false);
    param_set_compaction(&a.xcsf, true);
    param_set_explore(&a.xcsf, true);
    REQUIRE(a.ok());

    FILE *fp = fopen("temp_param.bin", "wb");
    REQUIRE(fp != NULL);
    const size_t w = param_save(&a.xcsf, fp);
    fclose(fp);

    ParamFixture b(1, 1, 1);
    fp = fopen("temp_param.bin", "rb");
    REQUIRE(fp != NULL);
    const size_t r = param_load(&b.xcsf, fp);
    fclose(fp);
    remove("temp_param.bin");

    CHECK_EQ(w, r);
    CHECK(b.ok());
    CHECK_EQ(strcmp(b.xcsf.population_file, a.xcsf.population_file), 0);
    CHECK_EQ(b.xcsf.x_dim, a.xcsf.x_dim);
    CHECK_EQ(b.xcsf.y_dim, a.xcsf.y_dim);
    CHECK_EQ(b.xcsf.n_actions, a.xcsf.n_actions);
    CHECK_EQ(b.xcsf.explore, a.xcsf.explore);
    CHECK_EQ(b.xcsf.POP_INIT, a.xcsf.POP_INIT);
    CHECK_EQ(b.xcsf.MAX_TRIALS, a.xcsf.MAX_TRIALS);
    CHECK_EQ(b.xcsf.PERF_TRIALS, a.xcsf.PERF_TRIALS);
    CHECK_EQ(b.xcsf.POP_SIZE, a.xcsf.POP_SIZE);
    CHECK_EQ(b.xcsf.LOSS_FUNC, a.xcsf.LOSS_FUNC);
    CHECK(b.xcsf.HUBER_DELTA == doctest::Approx(a.xcsf.HUBER_DELTA));
    CHECK(b.xcsf.GAMMA == doctest::Approx(a.xcsf.GAMMA));
    CHECK_EQ(b.xcsf.TELETRANSPORTATION, a.xcsf.TELETRANSPORTATION);
    CHECK(b.xcsf.P_EXPLORE == doctest::Approx(a.xcsf.P_EXPLORE));
    CHECK_EQ(b.xcsf.SET_SUBSUMPTION, a.xcsf.SET_SUBSUMPTION);
    CHECK_EQ(b.xcsf.THETA_SUB, a.xcsf.THETA_SUB);
    CHECK(b.xcsf.E0 == doctest::Approx(a.xcsf.E0));
    CHECK(b.xcsf.ALPHA == doctest::Approx(a.xcsf.ALPHA));
    CHECK(b.xcsf.NU == doctest::Approx(a.xcsf.NU));
    CHECK(b.xcsf.BETA == doctest::Approx(a.xcsf.BETA));
    CHECK(b.xcsf.DELTA == doctest::Approx(a.xcsf.DELTA));
    CHECK_EQ(b.xcsf.THETA_DEL, a.xcsf.THETA_DEL);
    CHECK(b.xcsf.INIT_FITNESS == doctest::Approx(a.xcsf.INIT_FITNESS));
    CHECK(b.xcsf.INIT_ERROR == doctest::Approx(a.xcsf.INIT_ERROR));
    CHECK_EQ(b.xcsf.M_PROBATION, a.xcsf.M_PROBATION);
    CHECK_EQ(b.xcsf.STATEFUL, a.xcsf.STATEFUL);
    CHECK_EQ(b.xcsf.COMPACTION, a.xcsf.COMPACTION);
}

TEST_CASE("PARAM: load rejects bad data")
{
    /* a zero length population_file string */
    {
        ParamFixture f;
        FILE *fp = fopen("temp_param_bad.bin", "wb");
        REQUIRE(fp != NULL);
        const size_t len = 0;
        fwrite(&len, sizeof(size_t), 1, fp);
        fclose(fp);
        fp = fopen("temp_param_bad.bin", "rb");
        REQUIRE(fp != NULL);
        CHECK_EQ(param_load(&f.xcsf, fp), 0);
        fclose(fp);
        CHECK_FALSE(f.ok());
    }
    /* an empty file: len reads as 0 */
    {
        ParamFixture f;
        FILE *fp = fopen("temp_param_bad.bin", "wb");
        REQUIRE(fp != NULL);
        fclose(fp);
        fp = fopen("temp_param_bad.bin", "rb");
        REQUIRE(fp != NULL);
        CHECK_EQ(param_load(&f.xcsf, fp), 0);
        fclose(fp);
        CHECK_FALSE(f.ok());
    }
    remove("temp_param_bad.bin");
}

TEST_CASE("PARAM: print")
{
    ParamFixture f;

    /* redirect stdout into a temp file for the duration of the call */
    FILE *tmp = tmpfile();
    REQUIRE(tmp != NULL);
    fflush(stdout);
    const int saved = dup(fileno(stdout));
    dup2(fileno(tmp), fileno(stdout));

    param_print(&f.xcsf);

    fflush(stdout);
    dup2(saved, fileno(stdout));
    close(saved);

    /* read back what was printed */
    rewind(tmp);
    char buf[8192];
    const size_t n = fread(buf, 1, sizeof(buf) - 1, tmp);
    buf[n] = '\0';
    fclose(tmp);

    CHECK(f.ok());
    CHECK(n > 0);
    cJSON *json = cJSON_Parse(buf);
    CHECK((json != NULL)); // output is valid JSON
    CHECK((cJSON_GetObjectItem(json, "x_dim") != NULL));
    cJSON_Delete(json);
}

TEST_CASE("PARAM: json import of malformed json")
{
    ParamFixture f;
    param_json_import(&f.xcsf, "{bad");
    CHECK_FALSE(f.ok());
}

TEST_CASE("PARAM: json import of component args")
{
    /* integer actions ignore args entirely: args present, no error */
    {
        ParamFixture f(3, 1, 2);
        const std::string s = std::string("{\"action\": {\"type\": \"") +
            action_type_as_string(ACT_TYPE_INTEGER) +
            "\", \"args\": {\"anything\": 1}}}";
        param_json_import(&f.xcsf, s.c_str());
        CHECK(f.ok());
        CHECK_EQ(f.xcsf.act->type, ACT_TYPE_INTEGER);
    }

    /* neural action with an unknown arg: error names the parameter */
    {
        ParamFixture f(3, 1, 2);
        const std::string s = std::string("{\"action\": {\"type\": \"") +
            action_type_as_string(ACT_TYPE_NEURAL) +
            "\", \"args\": {\"layer_0\": {\"bogus\": 1}}}}";
        param_json_import(&f.xcsf, s.c_str());
        CHECK_FALSE(f.ok());
        CHECK((strstr(f.xcsf.err_msg, "Invalid action parameter") != NULL));
    }

    /* condition: valid args, then an unknown arg */
    {
        ParamFixture f;
        param_json_import(&f.xcsf,
                          "{\"condition\": {\"type\": \"hyperrectangle_csr\","
                          " \"args\": {\"min\": 0.1, \"max\": 0.9}}}");
        CHECK(f.ok());
    }

    {
        ParamFixture f;
        param_json_import(&f.xcsf,
                          "{\"condition\": {\"type\": \"hyperrectangle_csr\","
                          " \"args\": {\"bogus\": 1}}}");
        CHECK_FALSE(f.ok());
        CHECK((strstr(f.xcsf.err_msg, "Invalid condition parameter") != NULL));
    }

    /* prediction: valid args, then an unknown arg */
    {
        ParamFixture f;
        param_json_import(&f.xcsf,
                          "{\"prediction\": {\"type\": \"nlms_linear\","
                          " \"args\": {\"eta\": 0.2}}}");
        CHECK(f.ok());
    }

    {
        ParamFixture f;
        param_json_import(&f.xcsf,
                          "{\"prediction\": {\"type\": \"nlms_linear\","
                          " \"args\": {\"bogus\": 1}}}");
        CHECK_FALSE(f.ok());
        CHECK((strstr(f.xcsf.err_msg, "Invalid prediction parameter") != NULL));
    }
}
