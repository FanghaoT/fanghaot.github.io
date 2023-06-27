---
title: A Demonstration of Script function in PSIM
author: <1>
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

1. Create a script file by clicking on **Script** in the menu and selecting **Script-Tool**.
2. Define the parameters for your simulation. For example:

    ```
    folder = "C:\buck\"
    f = 50k  // switching frequency
    Vin = 100
    L = 100u
    C = 1u
    R = 10
    ```

3. Use the Simulate function to run the simulation. The syntax for the Simulate function is as follows:

    ```
    Simulate(SchematicFilePath, SimviewFilePath, SimulationOptions, ReturnGraph)
    ```

    For example:

    ```
    Simulate(folder+"buck.psimsch", folder+"buck.smv", TotalTime=6m, PrintTime=0.1m, g1)
    ```

    In the above code, `folder+"buck.psimsch"` specifies the path to the PSIM schematic file, `folder+"buck.smv"` specifies the path to the Simview file, and `TotalTime=6m` and `PrintTime=0.1m` are simulation options. `g1` contains all the waveform information.

    `g1` is an array containing all the waveform information, in my case, 4 voltage waveforms are measured, so `g1 = {Time,Vg1,Vg2,Vin,Vout}`. Getting an index of one variable of the array is easy, such as `g1[0]` means the `time` in this case.

By following these steps and modifying the script to match your specific simulation setup, you can run a simulation using a script in PSIM.

## Step 2: Run Multiple Simulations by Script

The simple way of doing multiple simulations are copying the same code and define the parameters differently, and name the Simview file differently too. In case of too many scenarios are needed, the `while` function can help. For example if I wish to do the simulations with different 20 different `L` values.

    ```
    i=0;
    run = 10; //define number of runs/scenarios
    g2 = array(run)
    while(i<run)
    {
        L = 10u+i*10u //L varies based on 'i'
        C = 1u
        R = 10

        output = "buck"+i+".smv";
        Simulate(folder+"buck.psimsch", folder+output, TotalTime=6m, PrintTime=0.1m, g1);

        if(i==0)
        {
            g2[0]=g1[0] //Time
        }

        g2[i+1]=g1[4];      //Save "Vout" for every scenario

        SetCurveName(g2[i+1],"run_"+string(i+1));

        i++;
    }
    Graphwrite(folder+"combine.smv",g2); //save g2 as the combine.smv
    ```

## Step 3: Analyze the waveform

PSIM Script can do many analysis. In this demostration, only average and ripple of output voltage will be analyzed. Suppose g1 = {Time,Vg1,Vg2,Vin,Vout} in each simulation.

    ```
    out = Array(0);
    row = SizeOf(g1[0]);                                  // read number of rows
    count = 0;
	T = g1[0];                                           // obtain time column
    Vout = g1[4];                                           // obtain iL column
    flag=1;
    while (flag==1&&count<row>)
    {
        if (T[count] > 0.4e-3)                           // start checking after t = 0.4ms
		{
            flag=0
		}
        count++;
    }
    Vout_mea = Copy(Vout, count, -1); //  Copy from 0.4ms to the end
    V_max = max(Vout_mea);
    V_min = min(Vout_mea);
    V_ripple = V_max - V_min;                          // calculate ripple
    AddToArray(out, "Vout ripple = " + string(V_ripple));    // save to the out array

    file = folder + "output_vout_ripple.txt";
    FileWrite(file, out);                                    // write result to a file
    ```

Reference

`https://www.youtube.com/watch?v=pwVVyRkuo50`

`Tutorial documents and examples from PSIM`

