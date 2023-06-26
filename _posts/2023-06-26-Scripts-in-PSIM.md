---
title: A Demonstration of Script function in PSIM
author: FanghaoTian
date: 2023-06-26 21:20:00 +0200
categories: [tutorial, power electronics]
tags: [PSIM, power electronics, simulation]
---

# PSIM

PSIM (PowerSIM) is a simulation software tool used for designing, simulating, and analyzing power electronics and motor drive systems. It is widely used by engineers and researchers in the fields of power electronics.

## Script Function

The script function in PSIM refers to the capability of writing and executing scripts to automate tasks, customize simulations, and perform advanced operations beyond the standard graphical user interface (GUI) interactions. The scripting functionality in PSIM allows users to extend the capabilities of the software by writing scripts in the PSIM script language.

## Automation of Buck Converter Simulation

Here is a demonstration of using the script function in automating a buck converter simulation. The complete tutorial can be found in [help-->Tutorials-->How to use scripts function]. However, this page shows a simple example of a buck converter scenario for users to get started easily.

## Step 1: Run One Simulation by Script

To run a simulation using a script in PSIM, follow these steps:

Create a script file by clicking on **Script** in the menu and selecting **Script-Tool**. Define the parameters for your simulation. For example:
'''psim
folder = "C:\Users\Fanghao\Documents\ResearchTopic\20230626PSIM\"
f = 50k  // switching frequency
Vin = 100
L = 100u
C = 1u
R = 10
'''

Use the Simulate function to run the simulation. The syntax for the Simulate function is as follows:
'''psim
Simulate(SchematicFilePath, SimviewFilePath, SimulationOptions, ReturnGraph)
'''
For example:
'''psim
Simulate(folder+"buck.psimsch", folder+"buck.smv", TotalTime=6m, PrintTime=0.1m, g1)
'''

In the above code, [folder+"buck.psimsch"] specifies the path to the PSIM schematic file, [folder+"buck.smv"] specifies the path to the Simview file, and [TotalTime=6m] and [PrintTime=0.1m] are simulation options. [g1] contains all the waveform information.

By following these steps and modifying the script to match your specific simulation setup, you can run a simulation using a script in PSIM.

# PSIM

PSIM (PowerSIM) is a simulation software tool used for designing, simulating, and analyzing power electronics and motor drive systems. It is widely used by engineers and researchers in the fields of power electronics.

## Script Function

The script function in PSIM refers to the capability of writing and executing scripts to automate tasks, customize simulations, and perform advanced operations beyond the standard graphical user interface (GUI) interactions. The scripting functionality in PSIM allows users to extend the capabilities of the software by writing scripts in the PSIM script language.

## Automation of Buck Converter Simulation

Here is a demonstration of using the script function in automating a buck converter simulation. The complete tutorial can be found in [help-->Tutorials-->How to use scripts function]. However, this page shows a simple example of a buck converter scenario for users to get started easily.

## Step 1: Run One Simulation by Script

To run a simulation using a script in PSIM, follow these steps:

1. Create a script file by clicking on **Script** in the menu and selecting **Script-Tool**.
2. Define the parameters for your simulation. For example:

    ```psim
    folder = "C:\Users\Fanghao\Documents\ResearchTopic\20230626PSIM\"
    f = 50k  // switching frequency
    Vin = 100
    L = 100u
    C = 1u
    R = 10
    ```

3. Use the Simulate function to run the simulation. The syntax for the Simulate function is as follows:

    ```psim
    Simulate(SchematicFilePath, SimviewFilePath, SimulationOptions, ReturnGraph)
    ```

    For example:

    ```psim
    Simulate(folder+"buck.psimsch", folder+"buck.smv", TotalTime=6m, PrintTime=0.1m, g1)
    ```

    In the above code, `folder+"buck.psimsch"` specifies the path to the PSIM schematic file, `folder+"buck.smv"` specifies the path to the Simview file, and `TotalTime=6m` and `PrintTime=0.1m` are simulation options. `g1` contains all the waveform information.

By following these steps and modifying the script to match your specific simulation setup, you can run a simulation using a script in PSIM.

