{
    chart: {
        type: 'scatter',
        plotBorderWidth: 1,
        zooming: {
            type: 'xy'
        },
        panning: true,
        panKey: 'shift'
    },
    legend: {
        enabled: false
    },

    title: {
        align: 'left'
    },

    subtitle: {
        text: 'Source: Poll data aggregated from <a href="https://www.pollbludger.net/fed2025/bludgertrack/polldata.htm?">PollBludger.net</a>, media reports and other sources',
        fontSize: 10,
        align: 'left'
    },

    accessibility: {
        point: {
            valueDescriptionFormat: '{point.name}, Date: {point.x} Poll Result: {point.y}'
        }
    },

    xAxis: {
        title: {
            text:'Date',
            },
      endOnTick: true,
      showLastLabel: true,
      startOnTick: true,
      type: "datetime",
      tickLength: 0,
    },

    yAxis: {
        title: {
            text:'Voting Intention',
            },
      endOnTick: true,
      labels: {
        format: "{value} %",
      },
      showLastLabel: true,
      startOnTick: true,
      tickLength: 0,
      gridLineWidth: 0.0
    },

    plotOptions: {
        scatter: {
            marker: {
                radius: 8,
                states: {
                    hover: {
                        enabled: true,
                        lineColor: 'rgb(0,0,0)',
                        opacity: 1.0
                    },
                    inactive: {
                        enabled: true,
                        opacity: 0.25,
                    },
                }
            },
            states: {
                hover: {
                    marker: {
                        enabled: false,
                        fillOpacity: 1.0,
                    }
                },
                inactive: {
                    enabled: true,
                    fillOpacity: 0.25,
                },
            },
            jitter: {
                x: 0.5
            },
            tooltip: {
            pointFormatter: function() {
            return Highcharts.dateFormat('%A, %e %B %Y', this.x) + '<br/>Poll Result: ' + this.y + '%<br/>' + 'Organisation:' + this.id
                },
            },
        },
        line: {
            lineWidth: 10.0,
            shadow: {
                opacity: 0.8,
                width: 6.0,
            },
            states: {
            hover: {
                enabled: false
            },
            inactive: {
                enabled: false
            },
        },
        tooltip: {
            pointFormat: '{series.name} support: <b>{point.y}%',
            },

        },
    },


    }