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
 * @file cond_dummy.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2019--2021.
 * @brief Always-matching dummy condition functions.
 */

#include "cond_dummy.h"
#include "utils.h"

/**
 * @brief Dummy initialisation function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be initialised.
 */
void
cond_dummy_init(const struct XCSF *xcsf, struct Cl *c)
{
    (void) xcsf;
    (void) c;
}

/**
 * @brief Dummy free function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be initialised.
 */
void
cond_dummy_free(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
}

/**
 * @brief Dummy copy function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] dest Destination classifier.
 * @param [in] src Source classifier.
 */
void
cond_dummy_copy(const struct XCSF *xcsf, struct Cl *dest, const struct Cl *src)
{
    (void) xcsf;
    (void) dest;
    (void) src;
}

/**
 * @brief Dummy cover function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being covered.
 * @param [in] x Input state to cover.
 */
void
cond_dummy_cover(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    (void) xcsf;
    (void) c;
    (void) x;
}

/**
 * @brief Dummy update function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be updated.
 * @param [in] x Input state.
 * @param [in] y Truth/payoff value.
 */
void
cond_dummy_update(const struct XCSF *xcsf, const struct Cl *c, const double *x,
                  const double *y)
{
    (void) xcsf;
    (void) c;
    (void) x;
    (void) y;
}

/**
 * @brief Dummy match function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition to match.
 * @param [in] x Input state.
 * @return True.
 */
bool
cond_dummy_match(const struct XCSF *xcsf, const struct Cl *c, const double *x)
{
    (void) xcsf;
    (void) c;
    (void) x;
    return true;
}

/**
 * @brief Dummy crossover function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 First classifier whose condition is being crossed.
 * @param [in] c2 Second classifier whose condition is being crossed.
 * @return False.
 */
bool
cond_dummy_crossover(const struct XCSF *xcsf, const struct Cl *c1,
                     const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Dummy mutate function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is being mutated.
 * @return False.
 */
bool
cond_dummy_mutate(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
    return false;
}

/**
 * @brief Dummy general function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c1 Classifier whose condition is tested to be more general.
 * @param [in] c2 Classifier whose condition is tested to be more specific.
 * @return False.
 */
bool
cond_dummy_general(const struct XCSF *xcsf, const struct Cl *c1,
                   const struct Cl *c2)
{
    (void) xcsf;
    (void) c1;
    (void) c2;
    return false;
}

/**
 * @brief Dummy print function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be printed.
 */
void
cond_dummy_print(const struct XCSF *xcsf, const struct Cl *c)
{
    printf("%s\n", cond_dummy_json_export(xcsf, c));
}

/**
 * @brief Dummy size function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition size to return.
 * @return 0.
 */
double
cond_dummy_size(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
    return 0;
}

/**
 * @brief Dummy save function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return 0.
 */
size_t
cond_dummy_save(const struct XCSF *xcsf, const struct Cl *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}

/**
 * @brief Dummy load function.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be written.
 * @param [in] fp Pointer to the file to be written.
 * @return 0.
 */
size_t
cond_dummy_load(const struct XCSF *xcsf, struct Cl *c, FILE *fp)
{
    (void) xcsf;
    (void) c;
    (void) fp;
    return 0;
}

/**
 * @brief Returns a json formatted string representation of a dummy condition.
 * @param [in] xcsf XCSF data structure.
 * @param [in] c Classifier whose condition is to be returned.
 * @return String encoded in json format.
 */
const char *
cond_dummy_json_export(const struct XCSF *xcsf, const struct Cl *c)
{
    (void) xcsf;
    (void) c;
    cJSON *json = cJSON_CreateObject();
    cJSON_AddStringToObject(json, "type", "dummy");
    const char *string = cJSON_Print(json);
    cJSON_Delete(json);
    return string;
}
