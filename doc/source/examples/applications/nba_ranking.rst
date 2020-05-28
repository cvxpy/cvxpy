
TODO
====

-  how to post data? small data file?

Predicting NBA game wins
========================

In this notebook, we'll form a simple model to predict NBA game winners
based on which teams are playing, whether they are playing at home or
away, and which players are playing in the game and for how long.

The goal is to demonstrate implementing a simple (but non-standard)
regression model with CVXPY.

Imports and helper functions
============================

In the cell below, we define some helper functions to load the data,
manipulate it, format it, and plot the results.

The actual optimization code will be given in-line later in the
notebook.

.. code:: 

    import os  
    import json
    import datetime
    from collections import defaultdict
    import numpy as np
    import cvxpy as cvx
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.size': 14})
    
    class PlayerAppearance(object):
        __slots__= "name", "id", "minutes", 'team'
        
    class TeamAppearance(object):
        __slots__ = 'id', 'score', 'players', 'home', 'gameid'
    
    class Game(object):
        __slots__ = 'winner', 'loser', 'id', 'season'
        
    
    def get_game_data():
        '''Return a list of game objects from `gamestats` directory.'''
        games = []
    
        for fn in os.listdir('gamestats'):
             if fn.endswith('.json'):
                with open('gamestats/'+fn) as f:
                    a = json.load(f)
    
                    hometeamid = int(a['resultSets'][0]['rowSet'][0][6])
                    game = Game()
                    game.season = int(a['resultSets'][0]['rowSet'][0][8])
                    game.id = int(a['resultSets'][0]['rowSet'][0][2])
    
                    teams = [TeamAppearance(), TeamAppearance()]
                    for i, jteam in enumerate(a['resultSets'][1]['rowSet']):
                        teams[i].id = str(jteam[4])
                        teams[i].score = jteam[21]
                        teams[i].home = jteam[3] == hometeamid
                        teams[i].gameid = game.id
                        teams[i].players = []
    
                    for jplayer in a['resultSets'][4]['rowSet']:
                        if jplayer[8] == None or jplayer[8] == 0.0:
                            continue
                        player = PlayerAppearance()
                        player.name = str(jplayer[5])
                        player.id = jplayer[4]
                        minutes = jplayer[8].split(':')
                        minutes = float(minutes[0]) + float(minutes[1])/60.0
                        player.minutes = minutes
                        player.team = str(jplayer[2])
    
                        for team in teams:
                            if player.team == team.id:
                                team.players.append(player)
    
                    if teams[0].score > teams[1].score:
                        game.winner, game.loser = teams[0], teams[1]
                    else:
                        game.winner, game.loser = teams[1], teams[0]
    
                    games.append(game)
        return games
    
    def index_mappings(games):
        '''Return mappings between player and game identifiers and an indexed ordering.
        
        Puts teams and players appearing in `games` in an arbitrary order.
        This is the ordering which will be used to determine the relationship between
        matrix rows in the matrix-stuffing step.
        
        Returns:
        teams    --  ordered list of teams
                     (index -> team)
        players  --  ordered list of player pairs: (player name, unique player ID)
                     (index -> player)
        team2idx --  mapping team to index
                     (team -> index)
        pid2idx  --  mapping from unique player id to index
                     (player -> index)
        '''
        teams = set()
        players = set()
        for game in games:
            for team in game.winner, game.loser:
                teams.add(team.id)
                for player in team.players:
                    players.add((player.name,player.id))
                    
    
        teams = list(teams)
        players = list(players)
        
        team2idx = {t: i for i, t in enumerate(teams)}
        pid2idx = {t[1]: i for i, t in enumerate(players)}
        
        return teams, players, team2idx, pid2idx
    
    def split_games(games,p=0.3):
        '''Return games split into disjoint training and hold-out sets.'''
        
        games_holdout = np.random.choice(len(games),size=int(len(games)*p),replace=False)
        
        # training set
        games_t = [games[i] for i in range(len(games)) if i not in games_holdout]
        
        # hold-out set
        games_h = [games[i] for i in games_holdout]
        
        return games_t, games_h, 
    
    def data_mats(games, team2idx,home=False,pid2idx=[]):
        '''Stuff feature matrices for fitting model.
        
        Features included in the matrices depend on the function inputs.
        
        data_mats(games, team2idx) -- Team identities are the only features.
            That is, W_ij = 1 if team j won game i. 0 otherwise.
            
        data_mats(games, team2idx, home=True) -- Appends a column to W and L corresponding to
            whether the team was the home team.
            
        data_mats(games, team2idx,home=True,pid2idx=pid2idx) -- Appends columns
            to W and L corresponding to which players played and what fraction
            of 48 minutes they played in the game. That is, *if* j corresponds
            to a player column, then W_ij = t/48, where t is how many minutes
            player j (from the winning team) played in game i.
        '''
        m, n = len(games), len(team2idx)
        W, L = np.zeros((m,n)), np.zeros((m,n))
        
        Wh, Lh = np.zeros((m,1)), np.zeros((m,1))
        
        n = len(pid2idx)
        Wp, Lp = np.zeros((m,n)), np.zeros((m,n))
    
        for i, game in enumerate(games):
            for A, Ah, Ap, team in [(W, Wh, Wp, game.winner),(L, Lh, Lp, game.loser)]:
                A[i,team2idx[team.id]] = 1
                if team.home:
                    Ah[i] = 1
                if len(pid2idx) > 0:
                    for player in team.players:
                        Ap[i,pid2idx[player.id]] = player.minutes/48
                        
        if home:
            W = np.hstack([W, Wh])
            L = np.hstack([L, Lh])
        if len(pid2idx) > 0:
            W = np.hstack([W, Wp])
            L = np.hstack([L, Lp])
        
        return W, L
    
    def percent_correct(W,L,w):
        '''Return percentage correctly predicted for games (W,L) and model parameters w.
        
        W and L can be from training or hold-out data.
        '''
        m, n = W.shape
        return float(sum((W-L).dot(w) > 0))/m
    
    def reg_path(games,holdout,gammas,N, home=False, playertime=False):
        '''Plot a regularization path.
        
        For each of N times, partition games into a training
        and hold-out set, with `holdout` giving the hold-out proportion.
        For each gamma in gammas, train the model on the training set
        and evaluate it on both the training and hold out sets.
        
        Plot a data point for each instance with gamma on the x-axis
        and the percent correct on the y-axis. Also plot a line showing
        the average for each gamma.
        
        Optional arguments:
        home -- include home team feature
        playertime -- include player minutes played features
        '''
        v_t = []
        v_h = []
        
        teams, players, team2idx, pid2idx = index_mappings(games)
        if playertime is False:
            pid2idx = []
    
        for i in range(N):
            print '\nPartition {}'.format(i),
            games_t, games_h = split_games(games, holdout)
            W, L = data_mats(games_t, team2idx, home, pid2idx)
            Wh, Lh = data_mats(games_h, team2idx, home, pid2idx)
            for gamma in gammas:
                print '.',
                x = fit(W,L,gamma)
                v_t.append((gamma,percent_correct(W,L,x)))
                v_h.append((gamma,percent_correct(Wh,Lh,x)))
                
        alpha = .3
        fig, ax = plt.subplots(2, 1, sharex=True)
        fig.set_size_inches((10,10))
        a,b = zip(*v_t)
        ax[0].scatter(a,b,alpha=alpha)
        ax[0].set_ylabel('Training set % correct')
        a,b = zip(*v_h)
        ax[1].scatter(a,b,alpha=alpha)
        ax[1].set_ylabel('Test set % correct')
        ax[1].set_xlabel('Regularization $\gamma$')
    
        for i, path in enumerate([v_t,v_h]):
            avg = defaultdict(float)
            tot = defaultdict(float)
            for gamma, pred in path:
                avg[gamma] += pred
                tot[gamma] += 1.0
    
            for gamma in avg.keys():
                avg[gamma] /= tot[gamma]
    
            gammas = sorted(avg.keys())
            vals = [avg[key] for key in gammas]
            ax[i].plot(gammas, vals, 'r' )
    
        ax[0].legend(('average','data partition instance'))
        
        return v_t, v_h

Raw data
========

Below, we load the NBA game data from the 'gamestats' folder. We have
games from NBA seasons 2010-2013. For each game, we identify the winning
and losing team and the NBA season in which the game was played. For
each team appearance in a game, we have the score, a list of player
appearances, and whether the team played at home or away. For each
player appearance in a game, we have the player's name, a unique player
ID, and the number of minutes they played in the game.

Model
=====

We will train our model on a set of :math:`m` games. The data point for
game :math:`i` consists of an ordered pair of feature vectors,
:math:`(x^w_i, x^l_i) \in \mathbf{R}^n \times \mathbf{R}^n`. The first
vector in the pair represents the winning team. Feature vectors consist
of :math:`n` features which may include which team is playing, if they
are the home team, and how many minutes each player on the team played.

We want to find model parameters :math:`w` for the classifier function
:math:`f_w(x,y) = (x-y)^T w`. The model predicts that a team with
features :math:`x` will beat a team with features :math:`y` if
:math:`f_w(x,y) > 0`. Note that the classification inequality is
homogenous in :math:`w`. That is, the classifier is invariant to
scalings of :math:`w`.

To properly predict all games, we would need
:math:`f_w(x^w_i, x^l_i) > 0` for all games :math:`i`. This set of
homogenous strict inequalities in :math:`w` is feasible if and only if
there is some scaling :math:`\tilde{w} = \alpha w` such that
:math:`f_{\tilde{w}}(x^w_i, x^l_i) \geq 1` for all :math:`i`. This
motivates the loss function which we'll use:

.. math::

   \begin{equation}
   L(w) = \frac{1}{m}\sum_{i=1}^m \max \lbrace 1 - (x^w_i - x^l_i)^T w, 0 \rbrace.
   \end{equation}

The function :math:`L(w)` assigns a positive loss whenever the
classification inequality :math:`f_w(x^w_i, x^l_i) \geq 1` is violated.

We'll choose :math:`w` by minimizing the loss :math:`L(w)`, plus a
regularization term :math:`\gamma \| w \|_2`. The regularization term is
used to prevent over-fitting.

We use the ``fit(W,L,gamma)`` function defined below to solve the
optimization problem

.. math::

   \begin{array}{ll}
   \mbox{minimize} & L(w) + \gamma \| w \|_2,
   \end{array}

where :math:`W,L \in \mathbf{R}^{m \times n}` are data matrices, with
row :math:`i` of :math:`W` (:math:`L`) corresponding to :math:`x^w_i`
(:math:`x^l_i`).

.. code:: 

    def fit(W,L,gamma):
        '''Return model parameters w trained on data (W,L) with regularization gamma.'''
        m,n = W.shape
        w = cvx.Variable(shape=(n,1))
    
        objective = cvx.sum(cvx.pos(1 - (W-L)*w))/m + gamma*cvx.norm(w)
        objective = cvx.Minimize( objective )
        prob = cvx.Problem(objective)
        result = prob.solve(solver=cvx.ECOS, verbose=False)
        if prob.status != 'optimal':
            print "ERROR!"
        return np.array(w.value).flatten()

Example
=======

Load the list of games from the ``gamestats`` directory and select only
the games from a single season.

.. code:: 

    games = get_game_data()
    
    # select just games in a single season
    games = [game for game in games if game.season == 2010]
    len(games)




.. parsed-literal::

    1230



Choose some ordering of the teams and players for indexing purposes, and
get a mapping between the index and unique identifiers.

.. code:: 

    teams, players, team2idx, pid2idx = index_mappings(games)

Randomly split the games so that 70% are in a training set and 30% are
in a hold-out set. Produce the data matrices :math:`W` and :math:`L` for
the training and hold-out sets. Use features giving the team id, if they
are the home team, and the time played for each player.

.. code:: 

    np.random.seed(2)
    games_t, games_h = split_games(games, .3)
    
    W, L = data_mats(games_t, team2idx, True, pid2idx)
    Wh, Lh = data_mats(games_h, team2idx, True, pid2idx)

Train the model with **no** regularization on the training set and
evaluate the classification performance on both the training and
hold-out set.

.. code:: 

    gamma = 0
    w = fit(W,L,gamma)
    
    print "Training set %% correct: %f"%percent_correct(W,L,w)
    print "Hold-out set %% correct: %f"%percent_correct(Wh,Lh,w)


.. parsed-literal::

    Training set % correct: 1.000000
    Hold-out set % correct: 0.582656


We can see that we've over fit the model because the classification is
perfect on the training set, but low on the hold-out set. Adding some
regularization should improve the predictive performance on the hold-out
set:

.. code:: 

    gamma = .03
    w = fit(W,L,gamma)
    
    print "Training set %% correct: %f"%percent_correct(W,L,w)
    print "Hold-out set %% correct: %f"%percent_correct(Wh,Lh,w)


.. parsed-literal::

    Training set % correct: 0.786295
    Hold-out set % correct: 0.682927


Regularization paths
====================

We'll find good choices for the regularization hyper-parameter
:math:`\gamma` by plotting a regularization path. We'll randomly
partition the data into training and hold-out sets and plot the
percentage of games predicted correctly over a range of choices of
:math:`\gamma`. We'll do this several times for each choice of
:math:`\gamma` and plot the average result.

We'll plot the regularization path over different sets of features.
First, we'll just use team identities, then we'll add whether the team
played at home, and then we'll add the times played of each player.

We should see an increase in predictive power as we add features.

Note that some of these examples may take a few minutes to run.

Team identities
---------------

When the features include just the team identities, the model reduces to
just assigning a ranking to each team. This model is so simple that it
is hard to over-fit it on our data, so regularization has little effect.

.. code:: 

    np.random.seed(0)
    gammas = np.linspace(0,1,20)
    v_t, v_h = reg_path(games,.3,gammas,5,False, False)


.. parsed-literal::

    
    Partition 0 . . . . . . . . . . . . . . . . . . . . 
    Partition 1 . . . . . . . . . . . . . . . . . . . . 
    Partition 2 . . . . . . . . . . . . . . . . . . . . 
    Partition 3 . . . . . . . . . . . . . . . . . . . . 
    Partition 4 . . . . . . . . . . . . . . . . . . . .



.. image:: nba_ranking_files/nba_ranking_15_1.png


Identities + home/away
----------------------

We add a single feature denoting whether the teams played at home. This
doesn't add much to our model complexity, so it is still difficult to
over-fit. The predictive power is about the same as the previous model.

.. code:: 

    np.random.seed(0)
    gammas = np.linspace(0,.1,50)
    v_t, v_h = reg_path(games,.3,gammas,5,True, False)


.. parsed-literal::

    
    Partition 0 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    Partition 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    Partition 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    Partition 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    Partition 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .



.. image:: nba_ranking_files/nba_ranking_17_1.png


Identities + home/away + player minutes
---------------------------------------

We add in the number of minutes each player played in each game. This
adds many variables to the model and thus makes it much easier to
over-fit. We can see that the classification is perfect on the training
set, which suggests over fitting. We find that we get better performance
on the hold-out set with some added regularization.

.. code:: 

    np.random.seed(0)
    gammas = np.linspace(0,.2,40)
    v_t, v_h = reg_path(games,.3,gammas,5,True, True)


.. parsed-literal::

    
    Partition 0 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    Partition 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    Partition 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    Partition 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    Partition 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .



.. image:: nba_ranking_files/nba_ranking_19_1.png

