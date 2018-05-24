
from itertools import chain, combinations
from aimacode.planning import Action
from aimacode.utils import expr

from layers import BaseActionLayer, BaseLiteralLayer, makeNoOp, make_node


class ActionLayer(BaseActionLayer):

    def _inconsistent_effects(self, actionA, actionB):
        """ Return True if an effect of one action negates an effect of the other

        See Also
        --------
        layers.ActionNode
        """
        # TODO: implement this function

        for effectA in actionA.effects:
            for effectB in actionB.effects:
                if effectA == ~effectB:
                    return True
        for effectB in actionB.effects:
            for effectA in actionA.effects:
                if effectB == ~effectA:
                    return True
        return False


    def _interference(self, actionA, actionB):
        """ Return True if the effects of either action negate the preconditions of the other 
        
        See Also
        --------
        layers.ActionNode
        """
        # TODO: implement this function
        for effectA in actionA.effects:
            for precondB in actionB.preconditions:
                if effectA == ~precondB:
                    return True
        for effectB in actionB.effects:
            for precondA in actionA.preconditions:
                if effectB == ~precondA:
                    return True
        return False

    def _competing_needs(self, actionA, actionB):
        """ Return True if any preconditions of the two actions are pairwise mutex in the parent layer
        
        See Also
        --------
        layers.ActionNode
        layers.BaseLayer.parent_layer
        """
        # TODO: implement this function
        for precondA in actionA.preconditions:
            for precondB in actionB.preconditions:
                if self.parent_layer.is_mutex(precondA,precondB):
                    return True
        return False


class LiteralLayer(BaseLiteralLayer):

    def _inconsistent_support(self, literalA, literalB):
        """ Return True if all ways to achieve both literals are pairwise mutex in the parent layer

        See Also
        --------
        layers.BaseLayer.parent_layer
        """
        # Go through all the layers and collect all the literals for A and B
        allLiteralA = []
        allLiteralB = []
        # print("items: ", self.parent_layer.children.items())
        # print("literalA: {} and literalB: {}".format(literalA, literalB))
        # Since the layers are a dictionary, need to split it up as key/value to better parse. A layer consist of a fluent and an action
        for action, fluents in self.parent_layer.children.items():
            # print("these are the fluents:", fluents)
            # print("these are the actions:", action)
            if literalA in fluents:
                allLiteralA.append(action)
            if literalB in fluents:
                allLiteralB.append(action)
        # print("allLiteralA:{} and allLiteralB:{}".format(allLiteralA, allLiteralB))
        for litA in allLiteralA:
            for litB in allLiteralB:
                if not self.parent_layer.is_mutex(litA, litB):
                    return False
        return True

    def _negation(self, literalA, literalB):
        """ Return True if two literals are negations of each other """
        # TODO: implement this function
        return literalA == ~literalB


class PlanningGraph:
    def __init__(self, problem, state, serialize=True, ignore_mutexes=False):
        """
        Parameters
        ----------
        problem : PlanningProblem
            An instance of the PlanningProblem class

        state : tuple(bool)
            An ordered sequence of True/False values indicating the literal value
            of the corresponding fluent in problem.state_map

        serialize : bool
            Flag indicating whether to serialize non-persistence actions. Actions
            should NOT be serialized for regression search (e.g., GraphPlan), and
            _should_ be serialized if the planning graph is being used to estimate
            a heuristic
        """
        self._serialize = serialize
        self._is_leveled = False
        self._ignore_mutexes = ignore_mutexes
        self.goal = set(problem.goal)

        # make no-op actions that persist every literal to the next layer
        no_ops = [make_node(n, no_op=True) for n in chain(*(makeNoOp(s) for s in problem.state_map))]
        self._actionNodes = no_ops + [make_node(a) for a in problem.actions_list]
        
        # initialize the planning graph by finding the literals that are in the
        # first layer and finding the actions they they should be connected to
        literals = [s if f else ~s for f, s in zip(state, problem.state_map)]
        layer = LiteralLayer(literals, ActionLayer(), self._ignore_mutexes)
        layer.update_mutexes()
        self.literal_layers = [layer]
        self.action_layers = []

    def LevelCost(self, graph, goal):
        """ Calculates the cost of a level. The level cost of a goal is equal to the level number of the first literal layer in the planning graph where the goal literal appears.

        :param graph: PlanningGraph
        :param goal: PlanningProblem.goal
        :return: a value corresponding to the cost of the level

        function LevelCost(graph, goal) returns a value
         inputs:
          graph, a leveled planning graph
          goal, a literal that is a goal in the planning graph

         for each layeri in graph.literalLayers do
          if goal in layeri then return i
        """
        cost = 0
        for literalLayer in graph.literal_layer:
            if goal in literalLayer:
                return cost
            cost += 1

    def h_levelsum(self):
        """ Calculate the level sum heuristic for the planning graph

        The level sum is the sum of the level costs of all the goal literals
        combined. The "level cost" to achieve any single goal literal is the
        level at which the literal first appears in the planning graph. Note
        that the level cost is **NOT** the minimum number of actions to
        achieve a single goal literal.
        
        For example, if Goal_1 first appears in level 0 of the graph (i.e.,
        it is satisfied at the root of the planning graph) and Goal_2 first
        appears in level 3, then the levelsum is 0 + 3 = 3.

        Hints
        -----
          - See the pseudocode folder for help on a simple implementation
          - You can implement this function more efficiently than the
            sample pseudocode if you expand the graph one level at a time
            and accumulate the level cost of each goal rather than filling
            the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)
        """
        # TODO: implement this function
        # i = 0
        # level_costs = []
        # costs = []
        # while not self._is_leveled:
        #     allGoalsMet = True
        #     for literal in self.literal_layers[-1]:  # literal_layers[-1] is the parent layer
        #         if literal not in costs and literal in self.goal:
        #             costs.append(literal)
        #             level_costs.append(i)
        #             i += 1
        #         self._extend()
        # return sum(level_costs)

        heu_level_sum = 0

        while True:
            last_layer = self.literal_layers[-1]
            # print("this is a goal: ", self.goal)
            for literal in last_layer:
                # print("this is a literal of the current layer: ", literal)
                # print("current level sum is: ", heu_level_sum)
                if literal in self.goal:
                    heu_level_sum += len(last_layer)

            if self._is_leveled:
                break
            self._extend()

        # print("heu_level_sum: ", heu_level_sum)
        return heu_level_sum

    def h_maxlevel(self):
        """ Calculate the max level heuristic for the planning graph

        The max level is the largest level cost of any single goal fluent.
        The "level cost" to achieve any single goal literal is the level at
        which the literal first appears in the planning graph. Note that
        the level cost is **NOT** the minimum number of actions to achieve
        a single goal literal.

        For example, if Goal1 first appears in level 1 of the graph and
        Goal2 first appears in level 3, then the levelsum is max(1, 3) = 3.

        Hints
        -----
          - See the pseudocode folder for help on a simple implementation
          - You can implement this function more efficiently if you expand
            the graph one level at a time until the last goal is met rather
            than filling the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)

        Notes
        -----
        WARNING: you should expect long runtimes using this heuristic with A*
        """
        # TODO: implement maxlevel heuristic
        # i = 0
        # level_costs = []
        # costs = []
        # while not self._is_leveled:
        #     allGoalsMet = True
        #     for literal in self.literal_layers[-1]: #literal_layers[-1] is the parent layer
        #         if literal not in costs and literal in self.goal:
        #             costs.append(literal)
        #             level_costs.append(i)
        #             i += 1
        #         self._extend()
        # return max(level_costs)

        heu_level_max = []

        while True:
            last_layer = self.literal_layers[-1]
            # print("this is a goal: ", self.goal)
            for literal in last_layer:
                # print("this is a literal of the current layer: ", literal)
                if literal in self.goal:
                    heu_level_max.append(len(last_layer))

            if self._is_leveled:
                break
            self._extend()
        # print("heu_level_max ",heu_level_max)
        return max(heu_level_max)

    def h_setlevel(self):
        """ Calculate the set level heuristic for the planning graph

        The set level of a planning graph is the first level where all goals
        appear such that no pair of goal literals are mutex in the last
        layer of the planning graph.

        Hints
        -----
          - See the pseudocode folder for help on a simple implementation
          - You can implement this function more efficiently if you expand
            the graph one level at a time until you find the set level rather
            than filling the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)

        Notes
        -----
        WARNING: you should expect long runtimes using this heuristic on complex problems
        """
        while True:
            last_layer = self.literal_layers[-1]
            if self.goal.issubset(last_layer):
                no_pair_mutex = True
                for goal1 in self.goal:
                    for goal2 in self.goal:
                        if last_layer.is_mutex(goal1, goal2):
                            no_pair_mutex = False
                            break
                if no_pair_mutex:
                    return len(self.literal_layers)-1
            if self._is_leveled:
                break
            self._extend()
        return len(self.literal_layers)-1

    ##############################################################################
    #                     DO NOT MODIFY CODE BELOW THIS LINE                     #
    ##############################################################################

    def fill(self, maxlevels=-1):
        """ Extend the planning graph until it is leveled, or until a specified number of
        levels have been added

        Parameters
        ----------
        maxlevels : int
            The maximum number of levels to extend before breaking the loop. (Starting with
            a negative value will never interrupt the loop.)

        Notes
        -----
        YOU SHOULD NOT THIS FUNCTION TO COMPLETE THE PROJECT, BUT IT MAY BE USEFUL FOR TESTING
        """
        while not self._is_leveled:
            if maxlevels == 0: break
            self._extend()
            maxlevels -= 1
        return self

    def _extend(self):
        """ Extend the planning graph by adding both a new action layer and a new literal layer

        The new action layer contains all actions that could be taken given the positive AND
        negative literals in the leaf nodes of the parent literal level.

        The new literal layer contains all literals that could result from taking each possible
        action in the NEW action layer. 
        """
        if self._is_leveled: return

        parent_literals = self.literal_layers[-1]
        parent_actions = parent_literals.parent_layer
        action_layer = ActionLayer(parent_actions, parent_literals, self._serialize, self._ignore_mutexes)
        literal_layer = LiteralLayer(parent_literals, action_layer, self._ignore_mutexes)

        for action in self._actionNodes:
            # actions in the parent layer are skipped because are added monotonically to planning graphs,
            # which is performed automatically in the ActionLayer and LiteralLayer constructors
            if action not in parent_actions and action.preconditions <= parent_literals:
                action_layer.add(action)
                literal_layer |= action.effects

                # add two-way edges in the graph connecting the parent layer with the new action
                parent_literals.add_outbound_edges(action, action.preconditions)
                action_layer.add_inbound_edges(action, action.preconditions)

                # # add two-way edges in the graph connecting the new literaly layer with the new action
                action_layer.add_outbound_edges(action, action.effects)
                literal_layer.add_inbound_edges(action, action.effects)

        action_layer.update_mutexes()
        literal_layer.update_mutexes()
        self.action_layers.append(action_layer)
        self.literal_layers.append(literal_layer)
        self._is_leveled = literal_layer == action_layer.parent_layer
