{

    chart: {
        type: 'boxplot',
        inverted: true,
        zooming: {
            type: 'xy'
        },
    },



    title: {
        align: 'left'
    },

    subtitle: {
        text: 'Source: Poll data aggregated from <a href="https://www.pollbludger.net/fed2025/bludgertrack/polldata.htm?">PollBludger.net</a>, media reports and other sources',
        fontSize: 10,
        align: 'left'
    },

    xAxis: {
        categories: [
            'ALP', 'LNP', 'GRN'
        ]
    },

    yAxis: {
        title: {
            text: 'Vote (%)'
        }
    },

    tooltip: {
        valueSuffix: ''
    },

    plotOptions: {
        columnrange: {
            borderRadius: '10%',
            dataLabels: {
                enabled: true,
                format: '{y}%'
            }
        }
    },

    legend: {
        enabled: false
    },

    tooltip: {
        headerFormat: '{point.key}<br/>',
        pointFormat: 'Median: {point.median}%<br/>95% confidence interval is<br/>{point.q1}% to {point.q3}%'
    },
    plotOptions: {
        boxplot: {

            lineWidth: 3,
            medianColor: '#FFFFFF',
            medianWidth: 5,
            stemWidth: 5,
            whiskerLength: '30%',
            whiskerWidth: 5
        }
    },
    

}