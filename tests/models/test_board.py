import pytest

from catanatron.models.map import MINI_MAP_TEMPLATE, CatanMap
from catanatron.models.enums import RESOURCES
from catanatron.models.board import Board, get_node_distances
from catanatron.models.player import Color


def test_initial_build_phase_bypasses_restrictions():
    board = Board()
    with pytest.raises(ValueError):  # not connected and not initial-placement
        board.build_settlement(Color.RED, 3)
    with pytest.raises(ValueError):  # not connected to settlement
        board.build_road(Color.RED, (3, 2))

    board.build_settlement(Color.RED, 3, initial_build_phase=True)


def test_roads_must_always_be_connected():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)

    with pytest.raises(ValueError):  # not connected to settlement
        board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (3, 2))
    board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (3, 4))


def test_must_build_distance_two():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))

    with pytest.raises(ValueError):  # distance less than 2
        board.build_settlement(Color.BLUE, 4, initial_build_phase=True)
    board.build_settlement(Color.BLUE, 1, initial_build_phase=True)


def test_placements_must_be_connected():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))

    with pytest.raises(ValueError):  # distance less than 2 (even if connected)
        board.build_settlement(Color.RED, 2)
    with pytest.raises(ValueError):  # not connected
        board.build_settlement(Color.RED, 1)

    board.build_road(Color.RED, (2, 1))
    board.build_settlement(Color.RED, 1)


def test_city_requires_settlement_first():
    board = Board()
    with pytest.raises(ValueError):  # no settlement there
        board.build_city(Color.RED, 3)

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_city(Color.RED, 3)


def test_calling_the_edge_differently_is_not_a_problem():
    """Tests building on (0,0,0), East is the same as (1,-1,0), West"""
    pass


def test_get_ports():
    board = Board()
    ports = board.map.port_nodes
    for resource in RESOURCES:
        assert len(ports[resource]) == 2
    assert len(ports[None]) == 8


def test_node_distances():
    node_distances = get_node_distances()
    assert node_distances[2][3] == 1

    # Test are symmetric
    assert node_distances[0][3] == 3
    assert node_distances[3][0] == 3

    assert node_distances[3][9] == 2
    assert node_distances[3][29] == 4

    assert node_distances[34][32] == 2
    assert node_distances[31][45] == 11


# ===== Buildable nodes
def test_buildable_nodes():
    board = Board()
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_node_ids(Color.RED, initial_build_phase=True)
    assert len(nodes) == 54


def test_buildable_nodes_in_mini_map():
    board = Board(catan_map=CatanMap.from_template(MINI_MAP_TEMPLATE, "random"))
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_node_ids(Color.RED, initial_build_phase=True)
    assert len(nodes) == 24


def test_placing_settlement_removes_four_buildable_nodes():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_node_ids(Color.RED, initial_build_phase=True)
    assert len(nodes) == 50
    nodes = board.buildable_node_ids(Color.BLUE, initial_build_phase=True)
    assert len(nodes) == 50


def test_buildable_nodes_respects_distance_two():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)

    board.build_road(Color.RED, (3, 4))
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0

    board.build_road(Color.RED, (4, 5))
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 1
    assert nodes.pop() == 5


def test_cant_use_enemy_roads_to_connect():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))

    board.build_settlement(Color.BLUE, 1, initial_build_phase=True)
    board.build_road(Color.BLUE, (1, 2))
    board.build_road(Color.BLUE, (0, 1))
    board.build_road(Color.BLUE, (0, 20))  # north out of center tile

    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0

    nodes = board.buildable_node_ids(Color.BLUE)
    assert len(nodes) == 1


# ===== Buildable edges
def test_buildable_edges_simple():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 3


def test_buildable_edges_in_mini():
    board = Board(catan_map=CatanMap.from_template(MINI_MAP_TEMPLATE, "random"))
    board.build_settlement(Color.RED, 19, initial_build_phase=True)
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 2


def test_buildable_edges():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 4


def test_water_edge_is_not_buildable():
    board = Board()
    top_left_north_edge = 45
    board.build_settlement(Color.RED, top_left_north_edge, initial_build_phase=True)
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 2


# ===== Find connected components
def test_connected_components_empty_board():
    board = Board()
    components = board.find_connected_components(Color.RED)
    assert len(components) == 0


def test_one_connected_component():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))
    board.build_settlement(Color.RED, 1, initial_build_phase=True)
    board.build_road(Color.RED, (1, 2))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1

    board.build_road(Color.RED, (0, 1))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1


def test_two_connected_components():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1

    board.build_settlement(Color.RED, 1, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 2


def test_three_connected_components_bc_enemy_cut_road():
    board = Board()
    # Initial Building Phase of 2 players:
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))

    board.build_settlement(Color.BLUE, 15, initial_build_phase=True)
    board.build_road(Color.BLUE, (15, 4))
    board.build_settlement(Color.BLUE, 34, initial_build_phase=True)
    board.build_road(Color.BLUE, (34, 13))

    board.build_settlement(Color.RED, 1, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))

    # Extend road in a risky way
    board.build_road(Color.RED, (5, 0))
    board.build_road(Color.RED, (5, 16))

    # Plow </3
    board.build_road(Color.BLUE, (5, 4))
    board.build_settlement(Color.BLUE, 5)

    components = board.find_connected_components(Color.RED)
    assert len(components) == 3


def test_connected_components():
    board = Board()
    assert board.find_connected_components(Color.RED) == []

    # Simple test: roads stay at component, disconnected settlement creates new
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))
    assert len(board.find_connected_components(Color.RED)) == 1
    assert len(board.find_connected_components(Color.RED)[0]) == 2

    # This is just to be realistic
    board.build_settlement(Color.BLUE, 13, initial_build_phase=True)
    board.build_road(Color.BLUE, (13, 14))
    board.build_settlement(Color.BLUE, 37, initial_build_phase=True)
    board.build_road(Color.BLUE, (37, 14))

    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))
    assert len(board.find_connected_components(Color.RED)) == 2
    assert len(board.find_connected_components(Color.RED)[0]) == 2
    assert len(board.find_connected_components(Color.RED)[1]) == 2

    # Merging subcomponents
    board.build_road(Color.RED, (1, 2))
    assert len(board.find_connected_components(Color.RED)) == 1
    assert len(board.find_connected_components(Color.RED)[0]) == 4

    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, (4, 15))
    board.build_road(Color.RED, (15, 17))
    assert len(board.find_connected_components(Color.RED)) == 1

    # Enemy cutoff
    board.build_road(Color.BLUE, (14, 15))
    board.build_settlement(Color.BLUE, 15)
    assert len(board.find_connected_components(Color.RED)) == 2


def test_building_road_to_enemy_works_well():
    board = Board()

    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))
    board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (1, 0))

    # Test building towards enemy works well.
    assert len(board.find_connected_components(Color.RED)) == 1
    assert len(board.find_connected_components(Color.RED)[0]) == 3


def test_building_into_enemy_doesnt_merge_components():
    board = Board()

    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)
    board.build_settlement(Color.RED, 16, initial_build_phase=True)
    board.build_settlement(Color.RED, 6, initial_build_phase=True)
    board.build_road(Color.RED, (16, 5))
    board.build_road(Color.RED, (5, 0))
    board.build_road(Color.RED, (6, 1))
    board.build_road(Color.RED, (1, 0))
    assert len(board.find_connected_components(Color.RED)) == 2


def test_enemy_edge_not_buildable():
    board = Board()
    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)
    board.build_road(Color.BLUE, (0, 1))

    board.build_settlement(Color.RED, 2, initial_build_phase=True)
    board.build_road(Color.RED, (2, 1))
    buildable_edges = board.buildable_edges(Color.RED)
    assert len(buildable_edges) == 3


def test_many_buildings():
    board = Board()
    board.build_settlement(Color.ORANGE, 7, True)
    board.build_settlement(Color.ORANGE, 12, True)
    board.build_road(Color.ORANGE, (6, 7))
    board.build_road(Color.ORANGE, (7, 8))
    board.build_road(Color.ORANGE, (8, 9))
    board.build_road(Color.ORANGE, (8, 27))
    board.build_road(Color.ORANGE, (26, 27))
    board.build_road(Color.ORANGE, (9, 10))
    board.build_road(Color.ORANGE, (10, 11))
    board.build_road(Color.ORANGE, (11, 12))
    board.build_road(Color.ORANGE, (12, 13))
    board.build_road(Color.ORANGE, (13, 34))
    assert len(board.find_connected_components(Color.ORANGE)) == 1

    board.build_settlement(Color.WHITE, 30, True)
    board.build_road(Color.WHITE, (29, 30))
    board.build_road(Color.WHITE, (10, 29))
    board.build_road(Color.WHITE, (28, 29))
    board.build_road(Color.WHITE, (27, 28))
    board.build_settlement(Color.WHITE, 10)  # cut
    board.build_road(Color.WHITE, (30, 31))
    board.build_road(Color.WHITE, (31, 32))
    board.build_settlement(Color.WHITE, 32)
    board.build_road(Color.WHITE, (11, 32))
    board.build_road(Color.WHITE, (32, 33))
    board.build_road(Color.WHITE, (33, 34))
    board.build_settlement(Color.WHITE, 34)
    board.build_road(Color.WHITE, (34, 35))
    board.build_road(Color.WHITE, (35, 36))

    board.build_settlement(Color.WHITE, 41, True)
    board.build_city(Color.WHITE, 41)
    board.build_road(Color.WHITE, (41, 42))
    board.build_road(Color.WHITE, (40, 42))
    board.build_settlement(Color.WHITE, 27)  # cut

    assert len(board.find_connected_components(Color.WHITE)) == 2
    assert len(board.find_connected_components(Color.ORANGE)) == 3


# TODO: Test super long road, cut at many places, to yield 5+ component graph


# ===== A settlement on an opponent's road (upstream #391, #393; catan-arena 2026-10-19)


def _red_road_ending_at_orange_then_cut():
    """RED's road runs 10-11-12-3-4-5-0 into ORANGE's settlement at 0; BLUE cuts it at 12."""
    board = Board()
    board.build_settlement(Color.ORANGE, 0, True)
    board.build_road(Color.ORANGE, (0, 1))
    board.build_settlement(Color.RED, 10, True)
    for edge in ((10, 11), (11, 12), (12, 3), (3, 4), (4, 5), (5, 0)):
        board.build_road(Color.RED, edge)
    board.build_settlement(Color.BLUE, 14, True)
    board.build_road(Color.BLUE, (14, 13))
    board.build_road(Color.BLUE, (13, 12))
    board.build_settlement(Color.BLUE, 12)  # the cut
    return board


def test_a_cut_does_not_open_a_road_out_of_an_opponents_settlement():
    board = _red_road_ending_at_orange_then_cut()
    assert not any(0 in edge for edge in board.buildable_edges(Color.RED))
    with pytest.raises(ValueError):
        board.build_road(Color.RED, (0, 20))
    assert sorted(map(sorted, board.connected_components[Color.RED])) == [[3, 4, 5], [10, 11]]
    assert board.road_lengths[Color.RED] == 4  # 12-3-4-5-0: both ends at buildings still count


def test_more_cuts_after_the_first_do_not_crash():
    # Upstream #391: the old split left nodes in no component and a later cut
    # raised TypeError. Same board, then WHITE cuts at 4 and BLUE settles past it.
    board = _red_road_ending_at_orange_then_cut()
    board.build_settlement(Color.WHITE, 17, True)
    board.build_road(Color.WHITE, (17, 15))
    board.build_road(Color.WHITE, (15, 4))
    board.build_settlement(Color.WHITE, 4)
    assert sorted(map(sorted, board.connected_components[Color.RED])) == [[3], [5], [10, 11]]
    assert board.road_lengths[Color.RED] == 2


def test_a_settlement_on_the_end_of_a_road_blocks_it_and_a_later_cut_keeps_the_rest():
    # Upstream PR #393's example.
    board = Board()
    board.build_settlement(Color.ORANGE, 18, initial_build_phase=True)
    board.build_settlement(Color.WHITE, 22, initial_build_phase=True)
    board.build_settlement(Color.BLUE, 15, initial_build_phase=True)
    for edge in [(16, 18), (5, 16), (0, 5), (0, 1), (1, 6)]:
        board.build_road(Color.ORANGE, edge)  # ORANGE's road ends at node 6
    board.build_road(Color.WHITE, (22, 23))
    board.build_road(Color.WHITE, (6, 23))
    board.build_settlement(Color.WHITE, 6)  # WHITE settles on the end of it

    with pytest.raises(ValueError):
        board.build_road(Color.ORANGE, (6, 7))  # out of WHITE's settlement
    assert board.road_lengths[Color.ORANGE] == 5

    board.build_road(Color.BLUE, (4, 15))
    board.build_road(Color.BLUE, (4, 5))
    board.build_settlement(Color.BLUE, 5)  # cuts ORANGE in the middle
    assert sorted(map(sorted, board.connected_components[Color.ORANGE])) == [[0, 1], [16, 18]]
    assert board.road_lengths[Color.ORANGE] == 3  # 5-0-1-6, from one building to the other


def test_a_setup_settlement_on_the_end_of_an_opponents_first_road_blocks_it():
    # Seen in a random game (seed 43): BLUE's second settlement sat on the end
    # of ORANGE's first road; the setup phase never updated ORANGE's network,
    # and ORANGE later built a road out of BLUE's settlement.
    board = Board()
    board.build_settlement(Color.ORANGE, 0, initial_build_phase=True)
    board.build_road(Color.ORANGE, (0, 1))
    board.build_road(Color.ORANGE, (1, 6))  # stands in for a later road; 6 is two steps from 0
    board.build_settlement(Color.BLUE, 6, initial_build_phase=True)
    assert sorted(map(sorted, board.connected_components[Color.ORANGE])) == [[0, 1]]
    assert not any(6 in edge for edge in board.buildable_edges(Color.ORANGE))
    with pytest.raises(ValueError):
        board.build_road(Color.ORANGE, (6, 7))


def test_road_networks_match_a_rebuild_from_scratch_in_random_games():
    """After every move, each colour's network as kept move by move equals the
    one rebuilt from the board, and never contains an opponent's building."""
    from catanatron.game import Game
    from catanatron.models.player import RandomPlayer

    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    cuts = 0
    for seed in range(40):
        game = Game([RandomPlayer(c) for c in colors], seed=seed)
        while game.winning_color() is None and game.state.num_turns < 1000:
            action = game.play_tick().action
            board = game.state.board
            for color in colors:
                kept = {frozenset(c) for c in board.connected_components[color]}
                assert kept == {frozenset(c) for c in board._components(color)}, (seed, action)
                assert not any(board.is_enemy_node(n, color) for c in kept for n in c), (seed, action)
            if action.action_type.name == "BUILD_SETTLEMENT" and any(
                board.roads.get(e) not in (None, action.color) for e in board.buildable_subgraph.edges(action.value)
            ):
                cuts += 1
    assert cuts > 0  # the games did put settlements on opponents' roads
