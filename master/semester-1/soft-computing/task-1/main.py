import pandas as pd
import matplotlib.pyplot as plt


def opt_1(df, start, end):
    filtered_df = df.loc[(df['Year'] >= start) & (df['Year'] < end)]
    out = filtered_df['Genre'].value_counts()
    # stdout
    print(out)
    # graph
    out.plot(kind="bar")
    plt.show()


def opt_2(df):
    correlation = df["EU_Sales"].corr(df["NA_Sales"])
    # stdout
    print("=", correlation)


def opt_3(df, start, end):
    x_list = list()
    y_list = list()

    for y in range(start, end):
        filtered_df = df.loc[df['Year'] == y]
        c = filtered_df["EU_Sales"].corr(filtered_df["NA_Sales"])
        x_list.append(y)
        y_list.append(c)
        # stdout
        print(y, c)

    # graph
    plt.scatter(x_list, y_list)
    plt.show()


def opt_4(df):
    filtered_df = df.loc[df['Genre'] == "Sports"]
    filtered_df['diff'] = filtered_df['NA_Sales'] - filtered_df['EU_Sales']
    # stdout
    print("Max:",
          filtered_df["Name"].loc[filtered_df["diff"].idxmax()],
          filtered_df["diff"].max())
    print("Min:",
          filtered_df["Name"].loc[filtered_df["diff"].idxmin()],
          filtered_df["diff"].min())
    print("Prumer:\t\t\t", filtered_df["diff"].mean())
    print("Smerodatna odchylka:\t", filtered_df["diff"].std())


def opt_5(df):
    # pocty her v jednotlivych letech
    x_list = list()
    y_list = list()

    for y in range(int(df["Year"].min()), int(df["Year"].max()) + 1):
        filtered_df = df.loc[df['Year'] == y]
        c = filtered_df[filtered_df.columns[0]].count()
        x_list.append(y)
        y_list.append(c)
        # stdout
        print(y, c)

    # graph
    plt.bar(x_list, y_list)
    plt.show()


def opt_6(df):
    # zakladni statisticke udeje (max, min, avg, std) ve vybrane lokaci
    loc = None
    while True:
        opt = input("Vyber lokaci:\n1) EU\n2) NA\n3) JP\n")
        match opt:
            case "1":
                loc = "EU_Sales"
                break
            case "2":
                loc = "EU_Sales"
                break
            case "3":
                loc = "EU_Sales"
                break
            case _:
                print("Neplatna volba")
    # stdout
    print("Max:",
          df["Name"].loc[df[loc].idxmax()],
          df[loc].max())
    print("Min:",
          df["Name"].loc[df[loc].idxmin()],
          df[loc].min())
    print("Prumer:\t\t\t", df[loc].mean())
    print("Smerodatna odchylka:\t", df[loc].std())


def opt_7(df):
    # zastoupeni vydavatelu her
    total_cnt = df[df.columns[0]].count()

    out = list()
    for p in df["Publisher"].unique():
        filtered_df = df.loc[df["Publisher"] == p]
        cnt = filtered_df[filtered_df.columns[0]].count()
        out.append([p, cnt])
    out.sort(key=lambda x: x[1], reverse=True)

    c = int(input("Zadej pocet zobrazenych vysledku:"))
    c = 0 if c < 0 else c

    bar_x = list()
    bar_y = list()
    for i in range(c):
        bar_x.append(out[i][0])
        bar_y.append(out[i][1])
        total_cnt -= out[i][1]
        print(out[i][0], out[i][1])
    
    bar_x.append("Ostatni")
    bar_y.append(total_cnt)

    fig1, ax1 = plt.subplots()
    ax1.pie(bar_y, labels=bar_x, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.show()


def main():
    # nacteni dat ze souboru (lokalni/sitove)
    path = input("Zadej relativni cestu k souboru [vgsales.csv]:")
    if len(path) == 0:
        path = "vgsales.csv"
    df = pd.read_csv(path)

    # zpracovani dat
    while True:
        print("-------------------------------------------------------------\n",
              "Moznosti zpracovani [1-4]:\n",
              "1) Graf cetnosti zanru (>=1990 & < 2000)\n",
              "2) Korelacni koeficient mezi NA a EU\n",
              "3) Korelacni koeficient mezi NA a EU (od 1985 po 2010)\n",
              "4) Statisticke udeje z rozdilu v prodeji NA a EU (Sports)\n",
              "5) Pocty her v jednotlivych letech\n",
              "6) Zakladni statisticke udeje ve vybrane lokaci\n",
              "7) Zastoupeni vydavatelu her\n",
              "q) Ukoncit\n",
              "-------------------------------------------------------------\n")
        opt = input()
        match opt:
            case "1":
                # bod zadani c. 1
                opt_1(df, 1990, 2000)
            case "2":
                # bod zadani c. 2
                opt_2(df)
            case "3":
                # bod zadani c. 3
                opt_3(df, 1985, 2010)
            case "4":
                # bod zadani c. 4
                opt_4(df)
            case "5":
                opt_5(df)
            case "6":
                opt_6(df)
            case "7":
                opt_7(df)
            case "q":
                exit()
            case _:
                print("Neznama volba")
        input("(Pro pokracovani zmackni libovolnou klavesu)")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
