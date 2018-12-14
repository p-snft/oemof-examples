# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
General description:
---------------------
Example that illustrates how to use custom component `GenericCHP` can be used.
In this case it is used to model a motoric chp.

Installation requirements:
---------------------------
This example requires the latest version of oemof. Install by:

    pip install oemof

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from oemof.solph import (Bus, EnergySystem, Flow, Model, Sink, Source,
                         Transformer, NonConvex)
from oemof.solph.components import (GenericCHP)
from oemof.network import Node
from oemof.outputlib import processing, views


d = {'load': [0,  50,  75, 100],
     'P_el': [0, 125, 188, 250],
     'P_th': [0, 164, 222, 272],
     'P_Hs': [0, 319, 452, 576]}


df = pd.DataFrame(data=d).set_index('load')

# select periods
periods = 1000

# create an energy system
idx = pd.date_range('1/1/2017', periods=periods, freq='H')

results = []

demand_el = np.random.uniform(low=0, high=df['P_el'][100], size=periods)
demand_th = np.random.uniform(low=0, high=df['P_th'][100], size=periods)

for i in range(2):
    es = EnergySystem(timeindex=idx)
    Node.registry = es

    b_el = Bus(label="b_el")
    b_gas = Bus(label="b_gas")
    b_th = Bus(label="b_th")

    r_gas = Source(label='r_gas', outputs={b_gas: Flow(
        variable_costs=0, nominal_value=df['P_Hs'][100])})
    r_el = Source(label='r_el', outputs={b_el: Flow(
        variable_costs=999, nominal_value=df['P_el'][100])})
    r_th = Source(label='r_th', outputs={b_th: Flow(
        variable_costs=999, nominal_value=df['P_th'][100])})

    s_el = Sink(label='s_el', inputs={b_el: Flow(
        actual_value=demand_el, fixed=True, nominal_value=1)})
    s_th = Sink(label='s_th', inputs={b_th: Flow(
        actual_value=demand_th, fixed=True, nominal_value=1)})

    H_L_FG_share_max = 1 - (df["P_th"][100] + df["P_el"][100])/df["P_Hs"][100]
    H_L_FG_share_min = 1 - (df["P_th"][50] + df["P_el"][50])/df["P_Hs"][50]
    P_max_woDH = df["P_el"][100]
    P_min_woDH = df["P_el"][50]
    Eta_el_max_woDH = df["P_el"][100] / df["P_Hs"][100]
    Eta_el_min_woDH = df["P_el"][50] / df["P_Hs"][50]

    if i == 0:
        mchp = GenericCHP(
            label='mchp',
            fuel_input={
                b_gas: Flow(
                    H_L_FG_share_max=np.full(periods, H_L_FG_share_max),
                    H_L_FG_share_min=[
                        H_L_FG_share_min for p in range(0, periods)])},
            electrical_output={b_el: Flow(
                P_max_woDH=np.full(periods, P_max_woDH),
                P_min_woDH=np.full(periods, P_min_woDH),
                Eta_el_max_woDH=np.full(periods, Eta_el_max_woDH),
                Eta_el_min_woDH=np.full(periods, Eta_el_min_woDH))},
            heat_output={b_th: Flow(Q_CW_min=np.zeros(periods))},
            Beta=np.zeros(periods),
            back_pressure=False)
    else:
        bpt = GenericCHP(
            label='bpt',
            fuel_input={
                b_gas: Flow(
                    H_L_FG_share_max=np.full(periods, H_L_FG_share_max))},
            electrical_output={b_el: Flow(
                P_max_woDH=np.full(periods, P_max_woDH),
                P_min_woDH=np.full(periods, P_min_woDH),
                Eta_el_max_woDH=np.full(periods, Eta_el_max_woDH),
                Eta_el_min_woDH=np.full(periods, Eta_el_min_woDH))},
            heat_output={b_th: Flow(Q_CW_min=np.zeros(periods))},
            Beta=np.zeros(periods),
            back_pressure=True)

    # create an optimization problem and solve it
    om = Model(es)

    # debugging
    # om.write('generic_chp.lp', io_options={'symbolic_solver_labels': True})

    # solve model
    om.solve(solver='cbc', solve_kwargs={'tee': False})

    # create result object
    results.append(processing.results(om))

# plot data
if plt is not None:
    data_mchp = results[0][(mchp, None)]['sequences']
    data_bpt = results[1][(bpt, None)]['sequences']


    def p(x):
        y = np.where(
            x < df["P_Hs"][50], 0,
            np.where(
                x < df["P_Hs"][75],
                (x-df["P_Hs"][50]) * (df["P_el"][75]-df["P_el"][50]) / (df["P_Hs"][75]-df["P_Hs"][50]) + df["P_el"][50],
                (x - df["P_Hs"][75]) * (df["P_el"][100] - df["P_el"][75]) / (df["P_Hs"][100] - df["P_Hs"][75]) + df["P_el"][75]))
        return y

    def q(x):
        y = np.where(
            x < df["P_Hs"][50], 0,
            np.where(
                x < df["P_Hs"][75],
                (x - df["P_Hs"][50]) * (df["P_th"][75] - df["P_th"][50]) / (df["P_Hs"][75] - df["P_Hs"][50]) + df["P_th"][50],
                (x - df["P_Hs"][75]) * (df["P_th"][100] - df["P_th"][75]) / (df["P_Hs"][100] - df["P_Hs"][75]) + df["P_th"][75]))
        return y

    plt.plot(df['P_Hs'][1:], df['P_el'][1:], "bo-")
    plt.scatter(data_mchp.H_F, data_mchp.P, c="m", label="mCHP", marker="o")
    plt.scatter(data_bpt.H_F, data_bpt.P, c="c", label="BPT", marker="+")

    plt.plot(df['P_Hs'][1:], df['P_th'][1:], "ro-")
    plt.scatter(data_mchp.H_F, data_mchp.Q, c="m", label="mCHP", marker="o")
    plt.scatter(data_bpt.H_F, data_bpt.Q, c="c", label="BPT", marker="+")

    plt.grid()
    plt.xlabel(r"$P_{in,Hs}$ (kW)")
    plt.ylabel(r"$P_{out}$ (kW)")
    plt.legend()

    plt.figure()

    plt.scatter(data_mchp.H_F,
                100*(data_mchp.P-p(data_mchp.H_F))/p(data_mchp.H_F),
                c="b", label="P(mCHP)", marker="o")
    plt.scatter(data_mchp.H_F,
                100*(data_mchp.Q-q(data_mchp.H_F))/q(data_mchp.H_F),
                c="r", label="Q(mCHP)", marker="o")

    plt.scatter(data_bpt.H_F,
                100*(data_bpt.P-p(data_bpt.H_F))/p(data_bpt.H_F),
                c="b", label="P(BPT)", marker="x")
    plt.scatter(data_bpt.H_F,
                100*(data_bpt.Q-q(data_bpt.H_F))/q(data_bpt.H_F),
                c="r", label="Q(BPT)", marker="x")

    plt.xlim(300, 600)
    plt.ylim(-1, 1)

    plt.grid()
    plt.legend()

    plt.xlabel(r"$P_{in,Hs}$ (kW)")
    plt.ylabel(r"$\Delta P_{out}$ (%)")

    plt.figure()

    plt.scatter(q(data_bpt.H_F), p(data_bpt.H_F),
                c="k", label="piecewise", marker="x")
    plt.scatter(data_mchp.Q, data_mchp.P, c="m", label="mCHP", marker="o")
    plt.scatter(data_bpt.Q, data_bpt.P, c="c", label="BPT", marker="+")

    plt.grid()
    plt.xlabel(r"$Q$ (kW)")
    plt.ylabel(r"$P$ (kW)")
    plt.legend()

    plt.legend()

    plt.show()

