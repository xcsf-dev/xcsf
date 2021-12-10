#!/usr/bin/python3
#
# Copyright (C) 2021 Richard Preen <rpreen@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Classes for visualising classifier knowledge representations.
"""

from __future__ import annotations

from typing import List, Optional

import graphviz


class TreeViz:
    """Visualises a GP tree with graphviz."""

    def __init__(
        self,
        tree: List[str],
        filename: str,
        note: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Plots a tree with graphviz, saving to filename.
        Optionally accepts a string to be added as a note/caption.
        Optionally uses a list of feature names."""
        self.feature_names: Optional[List[str]] = feature_names
        self.tree: List[str] = tree
        self.cnt: int = 0
        self.pos: int = 0
        self.gviz = graphviz.Graph("G", filename=filename + ".gv")
        self.read_subexpr()
        if note is not None:
            self.gviz.attr(label=note)
        self.gviz.view()

    def label(self, symbol: str) -> str:
        """Returns the node label for a symbol."""
        if self.feature_names is not None and isinstance(symbol, str):
            start, end = symbol.split("_") if "_" in symbol else (symbol, "")
            if start == "feature" and int(end) < len(self.feature_names):
                return self.feature_names[int(end)]
        elif isinstance(symbol, float):
            return "%.5f" % symbol
        return str(symbol)

    def read_function(self) -> str:
        """Parses functions."""
        expr1: str = self.read_subexpr()
        symbol: str = self.tree[self.pos]
        if symbol in ("+", "-", "*", "/"):
            self.pos += 1
            expr2 = self.read_function()
            self.cnt += 1
            self.gviz.edge(str(self.cnt), expr1)
            self.gviz.edge(str(self.cnt), expr2)
            self.gviz.node(str(self.cnt), label=self.label(symbol))
            return expr2
        return expr1

    def read_subexpr(self) -> str:
        """Parses sub-expressions."""
        symbol: str = self.tree[self.pos]
        self.pos += 1
        if symbol == "(":
            self.read_function()
            self.pos += 1  # ')'
        else:
            self.cnt += 1
            self.gviz.node(str(self.cnt), label=self.label(symbol))
        return str(self.cnt)


class DGPViz:
    """Visualises a DGP graph with graphviz."""

    def __init__(
        self,
        graph: dict,
        filename: str,
        note: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Plots a DGP graph with graphviz, saving to filename.
        Optionally accepts a string to be added as a note/caption.
        Optionally uses a list of feature names."""
        self.feature_names: Optional[List[str]] = feature_names
        self.n: int = graph["n"]
        self.n_inputs: int = graph["n_inputs"]
        self.functions: List[str] = graph["functions"]
        self.connectivity: List[int] = graph["connectivity"]
        self.k: int = int(len(self.connectivity) / self.n)
        self.gviz = graphviz.Digraph("G", filename=filename + ".gv")
        self.draw()
        label: str = "" if note is None else note
        label += "\nN = %d\n" % graph["n"]
        label += "T = %d\n" % graph["t"]
        label += "match node shaded\n"
        self.gviz.attr(label=label)
        self.gviz.view()

    def label(self, symbol: str) -> str:
        """Returns the node label for a symbol."""
        if self.feature_names is not None and isinstance(symbol, str):
            start, end = symbol.split("_") if "_" in symbol else (symbol, "")
            if start == "feature" and int(end) < len(self.feature_names):
                return self.feature_names[int(end)]
        elif isinstance(symbol, float):
            return "%.5f" % symbol
        return str(symbol)

    def draw(self) -> None:
        """Plots the nodes and edges in the graph."""
        for i in range(self.n):
            style: str = "filled" if i == 0 else ""  # fill the match node
            self.gviz.node(str(i), label=self.functions[i], style=style)
            n_inputs: int = 1 if self.functions[i] == "Fuzzy NOT" else self.k
            for j in range(n_inputs):
                src = self.connectivity[(i * self.k) + j]
                if src < self.n_inputs:
                    feature = "feature_%d" % src
                    self.gviz.node(feature, label=self.label(feature), shape="square")
                    self.gviz.edge(feature, str(i))
                else:
                    self.gviz.edge(str(src - self.n_inputs), str(i))
