import pickle
import copy
from collections import defaultdict
from typing import Any, Set, Dict, Tuple, List
import functools

import networkx as nx  # type: ignore

from catanatron.models.player import Color
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    NUM_NODES,
    CatanMap,
    NodeId,
)
from catanatron.models.enums import FastBuildingType, SETTLEMENT, CITY


# Used to find relationships between nodes and edges
base_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
mini_map = CatanMap.from_template(MINI_MAP_TEMPLATE)
STATIC_GRAPH = nx.Graph()
for tile in base_map.tiles.values():
    STATIC_GRAPH.add_nodes_from(tile.nodes.values())
    STATIC_GRAPH.add_edges_from(tile.edges.values())


@functools.lru_cache(1)
def get_node_distances():
    return nx.floyd_warshall(STATIC_GRAPH)


@functools.lru_cache(3)  # None, range(54), range(24)
def get_edges(land_nodes=None):
    return list(STATIC_GRAPH.subgraph(land_nodes or range(NUM_NODES)).edges())


class Board:
    """Encapsulates all state information regarding the board.

    Attributes:
        buildings (Dict[NodeId, Tuple[Color, FastBuildingType]]): Mapping from
            node id to building (if there is a building there).
        roads (Dict[EdgeId, Color]): Mapping from edge
            to Color (if there is a road there). Contains inverted
            edges as well for ease of querying.
        connected_components (Dict[Color, List[Set[NodeId]]]): Cache
            datastructure to speed up maintaining longest road computation.
            To be queried by Color. Value is a list of node sets.
        board_buildable_ids (Set[NodeId]): Cache of buildable node ids in board.
        road_color (Color): Color of player with longest road.
        road_length (int): Number of roads of longest road
        robber_coordinate (Coordinate): Coordinate where robber is.
    """

    def __init__(self, catan_map=None, initialize=True):
        self.buildable_subgraph: Any = None
        self.buildable_edges_cache = {}
        self.player_port_resources_cache = {}
        if initialize:
            self.map: CatanMap = catan_map or CatanMap.from_template(
                BASE_MAP_TEMPLATE
            )  # Static State (no need to copy)

            self.buildings: Dict[NodeId, Tuple[Color, FastBuildingType]] = dict()
            self.roads = dict()  # (node_id, node_id) => color

            # color => int{}[] (list of node_id sets) one per component
            #   nodes in sets are incidental (might not be owned by player)
            self.connected_components: Any = defaultdict(list)
            self.board_buildable_ids = set(self.map.land_nodes)
            self.road_lengths = defaultdict(int)
            self.road_color = None
            self.road_length = 0
            # set_position: whether the holder above is KNOWN (the HUD's
            # badge read), inferred from the lengths, or unknown; and the
            # colours that could hold it when it is unknown (catan-player)
            self.road_holder_known = True
            self.road_holder_inferred = False
            self.road_candidates = []

            # assumes there is at least one desert:
            self.robber_coordinate = filter(
                lambda coordinate: self.map.land_tiles[coordinate].resource is None,
                self.map.land_tiles.keys(),
            ).__next__()

            # Cache buildable subgraph
            self.buildable_subgraph = STATIC_GRAPH.subgraph(self.map.land_nodes)

    def build_settlement(self, color, node_id, initial_build_phase=False):
        """Adds a settlement, and ensures is a valid place to build.

        Args:
            color (Color): player's color
            node_id (int): where to build
            initial_build_phase (bool, optional):
                Whether this is part of initial building phase, so as to skip
                connectedness validation. Defaults to True.
        """
        buildable = self.buildable_node_ids(
            color, initial_build_phase=initial_build_phase
        )
        if node_id not in buildable:
            raise ValueError(
                "Invalid Settlement Placement: not connected and not initial-placement"
            )

        if node_id in self.buildings:
            raise ValueError("Invalid Settlement Placement: a building exists there")

        self.buildings[node_id] = (color, SETTLEMENT)

        previous_road_color = self.road_color
        if initial_build_phase:
            self.connected_components[color].append({node_id})
        else:
            # Maybe cut connected components.
            edges_by_color = defaultdict(list)
            for edge in STATIC_GRAPH.edges(node_id):
                edges_by_color[self.roads.get(edge, None)].append(edge)

            for edge_color, edges in edges_by_color.items():
                if edge_color == color or edge_color is None:
                    continue  # ignore
                if len(edges) == 2:  # rip, edge_color has been plowed
                    # consider cut was at b=node_id for edges (a, b) and (b, c)
                    a = [n for n in edges[0] if n != node_id].pop()
                    c = [n for n in edges[1] if n != node_id].pop()

                    # do dfs from a adding all encountered nodes
                    a_nodeset = self.dfs_walk(a, edge_color)
                    c_nodeset = self.dfs_walk(c, edge_color)

                    # split this components on here.
                    b_index = self._get_connected_component_index(node_id, edge_color)
                    del self.connected_components[edge_color][b_index]
                    self.connected_components[edge_color].append(a_nodeset)
                    if c_nodeset != a_nodeset:
                        # On a loop, cutting one node disconnects nothing and both
                        # walks come back identical: there is no second component.
                        self.connected_components[edge_color].append(c_nodeset)

                    # Update longest road by plowed player. Compare again with all
                    self.road_lengths[edge_color] = max(
                        len(longest_acyclic_path(self, component, edge_color))
                        for component in self.connected_components[edge_color]
                    )
                    self._resolve_road_holder()
                elif len(edges) == 1:
                    # Endpoint blocked: remove node_id from opponent's component
                    b_index = self._get_connected_component_index(node_id, edge_color)
                    if b_index is not None:
                        self.connected_components[edge_color][b_index].discard(node_id)

        self.board_buildable_ids.discard(node_id)
        for n in STATIC_GRAPH.neighbors(node_id):
            self.board_buildable_ids.discard(n)

        self.buildable_edges_cache = {}  # Reset buildable_edges
        self.player_port_resources_cache = {}  # Reset port resources
        return previous_road_color, self.road_color, self.road_lengths

    def _resolve_road_holder(self):
        """Decide who holds the Longest Road card, per the official rules.

        `self.road_color = None` means the card is back in the bank.

        Both places that can change road lengths call this, so the rules live in
        one spot: build_road used to check the threshold itself while the cut
        branch of build_settlement just took an argmax, and the two disagreed.
        """
        best = max(self.road_lengths.values(), default=0)
        if best < 5:
            self.road_color, self.road_length = None, 0
            return

        leaders = [c for c, length in self.road_lengths.items() if length == best]
        if self.road_color in leaders:
            pass  # holder keeps the card when tied
        elif len(leaders) == 1:
            self.road_color = leaders[0]  # a single challenger takes it
        else:
            self.road_color = None  # tied challengers: card goes back to the bank
        self.road_length = best if self.road_color is not None else 0

    def dfs_walk(self, node_id, color):
        """Generates set of nodes that are "connected" to given node.

        Args:
            node_id (int): Where to start search/walk.
            color (Color): Player color asking

        Returns:
            Set[int]: Nodes that are "connected" to this one
                by roads of the color player.
        """
        agenda = [node_id]  # assuming node_id is owned.
        visited = set()

        while len(agenda) != 0:
            n = agenda.pop()
            visited.add(n)

            if self.is_enemy_node(n, color):
                continue  # end of the road

            neighbors = [v for v in STATIC_GRAPH.neighbors(n) if v not in visited]
            expandable = [v for v in neighbors if self.roads.get((n, v), None) == color]
            agenda.extend(expandable)

        return visited

    def _get_connected_component_index(self, node_id, color):
        for i, component in enumerate(self.connected_components[color]):
            if node_id in component:
                return i

    def build_road(self, color, edge):
        buildable = self.buildable_edges(color)
        inverted_edge = (edge[1], edge[0])
        if edge not in buildable and inverted_edge not in buildable:
            raise ValueError("Invalid Road Placement")

        self.roads[edge] = color
        self.roads[inverted_edge] = color

        # Find connected components corresponding to edge nodes (buildings).
        a, b = edge
        a_index = self._get_connected_component_index(a, color)
        b_index = self._get_connected_component_index(b, color)

        # Extend or merge components
        if a_index is None and not self.is_enemy_node(a, color):
            component = self.connected_components[color][b_index]
            component.add(a)
        elif b_index is None and not self.is_enemy_node(b, color):
            component = self.connected_components[color][a_index]
            component.add(b)
        elif a_index is not None and b_index is not None and a_index != b_index:
            # Merge both components into one and delete the other.
            component = set.union(
                self.connected_components[color][a_index],
                self.connected_components[color][b_index],
            )
            self.connected_components[color][a_index] = component
            del self.connected_components[color][b_index]
        else:
            # In this case, a_index == b_index, which means that the edge
            # is already part of one component. No actions needed.
            chosen_index = a_index if a_index is not None else b_index
            component = self.connected_components[color][chosen_index]

        # find longest path on component under question
        previous_road_color = self.road_color
        candidate_length = len(longest_acyclic_path(self, component, color))
        self.road_lengths[color] = max(self.road_lengths[color], candidate_length)
        self._resolve_road_holder()

        self.buildable_edges_cache = {}  # Reset buildable_edges
        return previous_road_color, self.road_color, self.road_lengths

    # ===== Loading a whole position (catan-player, 2026-09-17)
    #
    # build_settlement / build_road grow the board ONE PIECE AT A TIME and
    # assume the pieces arrive in the order they were built: an opponent's
    # settlement cuts a road network only if the roads were there first,
    # a road is refused unless it already connects. A position read off a
    # PICTURE has no order, and loading it through those two calls was the
    # root of three workarounds in catan-player (two-pass placement, the
    # _road_legal filter, roads dropped with a note when a later settlement
    # cut them). set_position writes every piece straight into the board
    # and rebuilds the derived state from scratch; validate says what is
    # wrong with the position instead of raising on the first piece.

    def set_position(self, buildings, roads, robber_coordinate=None,
                     longest_road_holder=None, longest_road_length=0, holder_known=True):
        """Load a whole position at once.

        Args:
            buildings: {node_id: (Color, SETTLEMENT | CITY)}
            roads: {(a, b): Color} -- either direction; both are stored
            robber_coordinate: the robber's tile, or None to leave it
            longest_road_holder: the Color the HUD shows holding Longest
                Road (None: nobody), and longest_road_length its length as
                the HUD shows it; the computed length wins when longer
            holder_known: False when the HUD's badge was NOT read (his ask,
                2026-09-17 01:00: "the system should explicitly say I don't
                know who has the longest road" -- None must not mean
                nobody). Then: one colour alone with the longest road of
                five or more holds it by the rules (inferred, said);
                two or more tied could each hold it, or nobody if the tie
                came from a cut -- the holder stays None, road_candidates
                lists them, validate() says so.
        Returns self. Nothing is checked here: call validate() after.
        """
        self.buildings = {int(n): (c, b) for n, (c, b) in buildings.items()}
        self.roads = {}
        for edge, color in roads.items():
            a, b = int(edge[0]), int(edge[1])
            self.roads[(a, b)] = color
            self.roads[(b, a)] = color

        # buildable nodes: land nodes not built on and not next to a building
        self.board_buildable_ids = set(self.map.land_nodes)
        for n in self.buildings:
            self.board_buildable_ids.discard(n)
            for m in STATIC_GRAPH.neighbors(n):
                self.board_buildable_ids.discard(m)

        # each colour's road networks, an opponent's building a dead end
        self.connected_components = defaultdict(list)
        colors = {c for c, _ in self.buildings.values()} | set(self.roads.values())
        for color in colors:
            self.connected_components[color] = self._components(color)

        # each colour's longest single line, and the award as the HUD shows it
        self.road_lengths = defaultdict(int)
        for color in colors:
            self.road_lengths[color] = max(
                (len(longest_acyclic_path(self, comp, color)) for comp in self.connected_components[color]),
                default=0,
            )
        longest = max(self.road_lengths.values(), default=0)
        self.road_holder_known = bool(holder_known) or longest_road_holder is not None
        self.road_holder_inferred = False
        self.road_candidates = sorted((c for c, n in self.road_lengths.items() if n == longest and n >= 5),
                                      key=lambda c: c.value) if longest >= 5 else []
        self.road_color = longest_road_holder
        if longest_road_holder is not None:
            self.road_length = max(int(longest_road_length or 0), self.road_lengths[longest_road_holder])
        else:
            self.road_length = longest if longest >= 5 else 0
            if not self.road_holder_known and len(self.road_candidates) == 1:
                self.road_color = self.road_candidates[0]        # alone at the top: the rules give it to them
                self.road_holder_inferred = True

        if robber_coordinate is not None:
            self.robber_coordinate = robber_coordinate
        self.buildable_edges_cache = {}
        self.player_port_resources_cache = {}
        return self

    def _components(self, color):
        """The connected pieces of `color`'s network, as build_road keeps
        them: every building node of the colour and every node its roads
        reach, walked along the colour's roads, an opponent's building
        being a dead end (never inside a component). A building with no
        road is a component of its own."""
        seeds = {n for n, (c, _) in self.buildings.items() if c == color}
        for (a, b), c in self.roads.items():
            if c == color:
                seeds.update((a, b))
        seeds = {n for n in seeds if not self.is_enemy_node(n, color)}
        comps = []
        left = set(seeds)
        while left:
            start = left.pop()
            comp = {start}
            agenda = [start]
            while agenda:
                n = agenda.pop()
                for v in STATIC_GRAPH.neighbors(n):
                    if v in comp or self.roads.get((n, v)) != color or self.is_enemy_node(v, color):
                        continue
                    comp.add(v)
                    agenda.append(v)
            left -= comp
            comps.append(comp)
        return sorted(comps, key=lambda s: min(s))

    def validate(self, expected=None, me=None, stock=(5, 4, 15)):
        """What is wrong with the loaded position, as findings rather than
        an exception: [(level, code, text)], level "critical" or "note".

        expected: {Color: {"vp": int, "longest_road": int,
                   "has_longest_road": bool, "has_largest_army": bool}}
                  -- the HUD's numbers, each optional
        me:       the Color asking; its own road island is critical, an
                  opponent's is a note
        stock:    settlements, cities, roads per seat
        """
        out = []
        # the distance rule: no two buildings on neighbouring nodes
        seen = set()
        for n, (c, b) in sorted(self.buildings.items()):
            for m in STATIC_GRAPH.neighbors(n):
                if m in self.buildings and (m, n) not in seen:
                    seen.add((n, m))
                    c2, b2 = self.buildings[m]
                    out.append(("critical", "adjacent",
                                "%s %s on node %d next to %s %s on node %d: the distance rule forbids it"
                                % (c.value, b.lower(), n, c2.value, b2.lower(), m)))
        # pieces against the stock
        counts = defaultdict(lambda: [0, 0, 0])
        for n, (c, b) in self.buildings.items():
            counts[c][0 if b == SETTLEMENT else 1] += 1
        for (a, b), c in self.roads.items():
            if a < b:
                counts[c][2] += 1
        for c, (s, ci, r) in counts.items():
            for have, limit, name in ((s, stock[0], "settlements"), (ci, stock[1], "cities"), (r, stock[2], "roads")):
                if have > limit:
                    out.append(("critical", "stock", "%s has %d %s on the board and the game gives %d" % (c.value, have, name, limit)))
        # road islands: a network touching no building of its colour and
        # ending at no opponent's building (a cut network is not an island)
        for color, comps in self.connected_components.items():
            for comp in comps:
                if any(self.buildings.get(n, (None, None))[0] == color for n in comp):
                    continue
                if any(self.roads.get((n, v)) == color and self.is_enemy_node(v, color)
                       for n in comp for v in STATIC_GRAPH.neighbors(n)):
                    continue
                edges = sorted({tuple(sorted((n, v))) for n in comp for v in STATIC_GRAPH.neighbors(n)
                                if self.roads.get((n, v)) == color})
                if not edges:
                    continue
                out.append(("critical" if color == me else "note", "island",
                            "%s road(s) %s touch no building of theirs" % (color.value, edges)))
        # the HUD's numbers against the pieces
        for color, e in (expected or {}).items():
            s, ci, _ = counts.get(color, (0, 0, 0))
            awards = 2 * int(self.road_color == color) + 2 * int(bool(e.get("has_largest_army")))
            visible = s + 2 * ci + awards
            vp = e.get("vp")
            if vp is not None:
                if int(vp) < visible:
                    out.append(("critical", "vp", "%s shows %d points on the HUD and %d on the board (%d settlements, %d cities, %d for awards)"
                                % (color.value, int(vp), visible, s, ci, awards)))
                elif int(vp) > visible:
                    out.append(("note", "hidden", "%s: %d point(s) not on the board (Victory Point cards)" % (color.value, int(vp) - visible)))
            lr = e.get("longest_road")
            if lr is not None and int(lr) != self.road_lengths.get(color, 0):
                out.append(("note", "road_length", "%s: the HUD says longest road %d, the pieces give %d"
                            % (color.value, int(lr), self.road_lengths.get(color, 0))))
        # the Longest Road holder: known, inferred, or unknown with the candidates
        cands = [c.value for c in getattr(self, "road_candidates", [])]
        if not getattr(self, "road_holder_known", True):
            if getattr(self, "road_holder_inferred", False):
                out.append(("note", "holder", "the HUD's Longest Road badge was not read: %s holds it by the rules "
                                              "(alone with a road of %d)" % (self.road_color.value, self.road_length)))
            elif len(cands) >= 2:
                # his call, 2026-09-17 01:15: "unknown tie is a stop"
                out.append(("critical", "holder", "the HUD's Longest Road badge was not read and %s each have a road of %d: "
                                                  "the holder is UNKNOWN -- one of them, or nobody if the tie came from a cut"
                                                  % (" and ".join(cands), self.road_length)))
        elif self.road_color is None and self.road_lengths:
            longest_color, longest = max(self.road_lengths.items(), key=lambda kv: kv[1])
            if longest >= 5:
                out.append(("note", "holder", "%s has a road of %d and the HUD shows no Longest Road holder"
                            % (longest_color.value, longest)))
        return out

    def build_city(self, color, node_id):
        building = self.buildings.get(node_id, None)
        if building is None or building[0] != color or building[1] != SETTLEMENT:
            raise ValueError("Invalid City Placement: no player settlement there")

        self.buildings[node_id] = (color, CITY)

    def buildable_node_ids(self, color: Color, initial_build_phase=False):
        if initial_build_phase:
            return sorted(list(self.board_buildable_ids))

        subgraphs = self.find_connected_components(color)
        nodes = set().union(*subgraphs)
        return sorted(list(nodes.intersection(self.board_buildable_ids)))

    def buildable_edges(self, color: Color):
        """List of (n1,n2) tuples. Edges are in n1 < n2 order."""
        if color in self.buildable_edges_cache:
            return self.buildable_edges_cache[color]

        expandable = set()

        # All nodes for this color.
        # TODO(tonypr): Explore caching for 'expandable_nodes'?
        # The 'expandable_nodes' set should only increase in size monotonically I think.
        # We can take advantage of that.
        expandable_nodes = set()
        expandable_nodes = expandable_nodes.union(*self.connected_components[color])

        candidate_edges = self.buildable_subgraph.edges(expandable_nodes)
        for edge in candidate_edges:
            if self.get_edge_color(edge) is None:
                expandable.add(tuple(sorted(edge)))

        self.buildable_edges_cache[color] = list(expandable)
        return self.buildable_edges_cache[color]

    def get_player_port_resources(self, color):
        """Yields resources (None for 3:1) of ports owned by color"""
        if color in self.player_port_resources_cache:
            return self.player_port_resources_cache[color]

        resources = set()
        for resource, node_ids in self.map.port_nodes.items():
            if any(self.is_friendly_node(node_id, color) for node_id in node_ids):
                resources.add(resource)

        self.player_port_resources_cache[color] = resources
        return resources

    def find_connected_components(self, color: Color):
        """
        Returns:
            nx.Graph[]: connected subgraphs. subgraphs
                might include nodes that color doesnt own (on the way and on ends),
                just to make it is "closed" and easier for buildable_nodes to operate.
        """
        return self.connected_components[color]

    def continuous_roads_by_player(self, color: Color):
        paths = []
        components = self.find_connected_components(color)
        for component in components:
            paths.append(longest_acyclic_path(self, component, color))
        return paths

    def copy(self):
        board = Board(self.map, initialize=False)
        board.map = self.map  # reuse since its immutable
        board.buildings = self.buildings.copy()
        board.roads = self.roads.copy()
        board.connected_components = pickle.loads(
            pickle.dumps(self.connected_components)
        )
        board.board_buildable_ids = self.board_buildable_ids.copy()
        board.road_lengths = self.road_lengths.copy()
        board.road_color = self.road_color
        board.road_length = self.road_length
        board.road_holder_known = getattr(self, "road_holder_known", True)
        board.road_holder_inferred = getattr(self, "road_holder_inferred", False)
        board.road_candidates = list(getattr(self, "road_candidates", []))

        board.robber_coordinate = self.robber_coordinate
        board.buildable_subgraph = self.buildable_subgraph
        board.buildable_edges_cache = copy.deepcopy(self.buildable_edges_cache)
        board.player_port_resources_cache = copy.deepcopy(
            self.player_port_resources_cache
        )
        return board

    # ===== Helper functions
    def get_node_color(self, node_id):
        # using try-except instead of .get for performance
        try:
            return self.buildings[node_id][0]
        except KeyError:
            return None

    def get_edge_color(self, edge):
        # using try-except instead of .get for performance
        try:
            return self.roads[edge]
        except KeyError:
            return None

    def is_enemy_node(self, node_id, color):
        node_color = self.get_node_color(node_id)
        return node_color is not None and node_color != color

    def is_enemy_road(self, edge, color):
        edge_color = self.get_edge_color(edge)
        return edge_color is not None and self.get_edge_color(edge) != color

    def is_friendly_node(self, node_id, color):
        return self.get_node_color(node_id) == color

    def is_friendly_road(self, edge, color):
        return self.get_edge_color(edge) == color


def longest_acyclic_path(board: Board, node_set: Set[int], color: Color):
    # catan-player, 2026-09-17: the counting fix of upstream PR #384 (broussal,
    # 2026-08-08, open) -- a road ending AT an opponent's building counts;
    # the route simply stops there. Upstream skipped the edge into an enemy
    # node, so every line ending at an opponent's settlement was one short
    # (the HUD showed one more than the engine on 30 of 99 recorded turns).
    # Two conditions are lifted: a route may FINISH on an enemy node, and
    # may START from one. `node_set` omits enemy nodes (never added by
    # build_road / set_position), so they would otherwise be unreachable as
    # DFS roots; a route has one start and one end, so two blocked
    # extremities consume one condition each.
    start_nodes = set(node_set)
    for node in node_set:
        for neighbor_node in STATIC_GRAPH.neighbors(node):
            if board.is_friendly_road(tuple(sorted((node, neighbor_node))), color):
                start_nodes.add(neighbor_node)

    paths = []
    for start_node in start_nodes:
        # do DFS when reach leaf node, stop and add to paths
        paths_from_this_node = []
        agenda: List[Tuple[int, Any]] = [(start_node, [])]
        while len(agenda) > 0:
            node, path_thus_far = agenda.pop()

            able_to_navigate = False
            for neighbor_node in STATIC_GRAPH.neighbors(node):
                edge = tuple(sorted((node, neighbor_node)))

                # Must travel on a friendly road.
                if not board.is_friendly_road(edge, color):
                    continue

                if edge in path_thus_far:
                    continue

                # Can't expand past an enemy node -- that would be two
                # consecutive segments meeting on it -- but the road leading
                # to it counts.
                if board.is_enemy_node(neighbor_node, color):
                    paths_from_this_node.append(path_thus_far + [edge])
                    continue

                agenda.append((neighbor_node, path_thus_far + [edge]))
                able_to_navigate = True

            if not able_to_navigate:  # then it is leaf node
                paths_from_this_node.append(path_thus_far)

        paths.extend(paths_from_this_node)

    return max(paths, key=len) if paths else []
