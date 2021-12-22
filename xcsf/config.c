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
 * @file config.c
 * @author Richard Preen <rpreen@gmail.com>
 * @copyright The Authors.
 * @date 2015--2021.
 * @brief Configuration file (JSON) handling functions.
 */

#include "config.h"
#include "param.h"

#define MAXLEN (127) //!< Maximum config file line length to read

/**
 * @brief Removes tabs/spaces/lf/cr
 * @param [in] s The line to trim.
 */
static void
config_trim(char *s)
{
    const char *d = s;
    do {
        while (*d == ' ' || *d == '\t' || *d == '\n' || *d == '\r') {
            ++d;
        }
    } while ((*s++ = *d++));
}

/**
 * @brief Reads the specified configuration file.
 * @param [in] xcsf The XCSF data structure.
 * @param [in] filename The name of the configuration file.
 */
void
config_read(struct XCSF *xcsf, const char *filename)
{
    FILE *f = fopen(filename, "rt");
    if (f == NULL) {
        printf("Warning: could not open %s. %s.\n", filename, strerror(errno));
        return;
    }
    fseek(f, 0, SEEK_END);
    const long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char file_buff[len];
    file_buff[0] = '\0';
    char line_buff[MAXLEN];
    while (!feof(f)) {
        if (fgets(line_buff, MAXLEN - 2, f) == NULL) {
            break;
        }
        config_trim(line_buff);
        if (strnlen(line_buff, MAXLEN) == 0 || line_buff[0] == '#') {
            continue; // ignore empty lines and lines starting with '#'
        }
        char *ptr = strchr(line_buff, '#'); // remove anything after #
        if (ptr != NULL) {
            *ptr = '\0';
        }
        strncat(file_buff, line_buff, MAXLEN);
    }
    fclose(f);
    param_json_import(xcsf, file_buff);
}
