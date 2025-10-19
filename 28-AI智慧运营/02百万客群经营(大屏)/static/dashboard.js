/**
 * 百万客群经营可视化大屏 JavaScript 组件
 * 提供数据展示、图表渲染和交互效果功能
 */

// 全局图表实例变量
let assetDistributionChart, lifecycleProductHeatmapChart, highPotentialCustomersChart, marketingPerformanceChart, geographicDistributionChart;

/**
 * 数字滚动动画效果
 * @param {string} id - 元素ID
 * @param {number} start - 起始值
 * @param {number} end - 结束值
 * @param {number} duration - 动画持续时间(毫秒)
 */
function animateValue(id, start, end, duration) {
    const obj = document.getElementById(id);
    if (!obj) return;
    
    const range = end - start;
    const minTimer = 50;
    let stepTime = Math.abs(Math.floor(duration / range));
    
    stepTime = Math.max(stepTime, minTimer);
    
    let startTime = new Date().getTime();
    let endTime = startTime + duration;
    let timer;
    
    function run() {
        let now = new Date().getTime();
        let remaining = Math.max((endTime - now) / duration, 0);
        let value = Math.round(end - (remaining * range));
        
        // 根据ID判断是否添加百分号
        if (id.includes('percent') || id.includes('success') || id.includes('active')) {
            obj.innerHTML = `${value}%`;
        } else {
            obj.innerHTML = value.toLocaleString();
        }
        
        if (value == end) {
            clearInterval(timer);
        }
    }
    
    timer = setInterval(run, stepTime);
    run();
}

/**
 * 进度条动画效果
 * @param {string} id - 元素ID
 * @param {number} value - 进度值(百分比)
 */
function animateProgress(id, value) {
    const progressBar = document.querySelector(`#${id} ~ .progress-bar .progress-fill`);
    if (progressBar) {
        setTimeout(() => {
            progressBar.style.width = `${value}%`;
        }, 300);
    }
}

/**
 * 初始化所有ECharts图表
 */
function initCharts() {
    try {
        assetDistributionChart = echarts.init(document.getElementById('assetDistributionChart'));
        lifecycleProductHeatmapChart = echarts.init(document.getElementById('lifecycleProductHeatmapChart'));
        highPotentialCustomersChart = echarts.init(document.getElementById('highPotentialCustomersChart'));
        marketingPerformanceChart = echarts.init(document.getElementById('marketingPerformanceChart'));
        geographicDistributionChart = echarts.init(document.getElementById('geographicDistributionChart'));
        
        // 响应式处理
        window.addEventListener('resize', function() {
            assetDistributionChart && assetDistributionChart.resize();
            lifecycleProductHeatmapChart && lifecycleProductHeatmapChart.resize();
            highPotentialCustomersChart && highPotentialCustomersChart.resize();
            marketingPerformanceChart && marketingPerformanceChart.resize();
            geographicDistributionChart && geographicDistributionChart.resize();
        });
    } catch(e) {
        console.error('Error initializing charts:', e);
    }
}

/**
 * 加载并渲染资产等级分布图表
 */
function loadAssetDistribution() {
    fetch('/api/asset_distribution')
        .then(response => response.json())
        .then(data => {
            document.getElementById('assetDistributionChart').classList.remove('loading');
            
            const option = {
                backgroundColor: 'transparent',
                tooltip: {
                    trigger: 'item',
                    formatter: '{a} <br/>{b}: {c} ({d}%)',
                    backgroundColor: 'rgba(26, 35, 50, 0.95)',
                    borderColor: '#4facfe',
                    borderWidth: 1,
                    textStyle: { color: '#fff' },
                    boxShadow: '0 0 15px rgba(79, 172, 254, 0.3)'
                },
                legend: {
                    orient: 'vertical',
                    left: 'left',
                    textStyle: { color: '#fff' },
                    top: 'middle',
                    itemWidth: 10,
                    itemHeight: 10,
                    itemGap: 15
                },
                series: [{
                    name: '客户数量',
                    type: 'pie',
                    radius: ['40%', '70%'], // 环形图
                    avoidLabelOverlap: false,
                    itemStyle: {
                        borderRadius: 10,
                        borderColor: '#1a2332',
                        borderWidth: 2,
                        color: function(params) {
                            const colorList = [
                                new echarts.graphic.LinearGradient(0, 0, 0, 1, [{offset: 0, color: '#4facfe'}, {offset: 1, color: '#00f2fe'}]),
                                new echarts.graphic.LinearGradient(0, 0, 0, 1, [{offset: 0, color: '#7b61ff'}, {offset: 1, color: '#a389ff'}]),
                                new echarts.graphic.LinearGradient(0, 0, 0, 1, [{offset: 0, color: '#00f2fe'}, {offset: 1, color: '#4facfe'}]),
                                new echarts.graphic.LinearGradient(0, 0, 0, 1, [{offset: 0, color: '#a389ff'}, {offset: 1, color: '#7b61ff'}])
                            ];
                            return colorList[params.dataIndex];
                        },
                        shadowBlur: 20,
                        shadowColor: 'rgba(79, 172, 254, 0.4)'
                    },
                    label: {
                        show: true,
                        formatter: '{b}: {c}',
                        color: '#fff',
                        fontSize: 12,
                        textShadowColor: 'rgba(79, 172, 254, 0.5)',
                        textShadowBlur: 5
                    },
                    emphasis: {
                        itemStyle: {
                            shadowBlur: 30,
                            shadowColor: 'rgba(79, 172, 254, 0.8)',
                            scale: 1.05
                        },
                        label: {
                            show: true,
                            fontSize: '16',
                            fontWeight: 'bold',
                            color: '#fff',
                            textShadowColor: 'rgba(79, 172, 254, 0.8)',
                            textShadowBlur: 10
                        }
                    },
                    data: data,
                    animationType: 'scale',
                    animationEasing: 'elasticOut',
                    animationDelay: function(idx) {
                        return Math.random() * 200;
                    },
                    animationDuration: 1500
                }]
            };
            
            assetDistributionChart.setOption(option);
        })
        .catch(error => console.error('Error fetching asset distribution:', error));
}

/**
 * 加载并渲染生命周期与产品持有热力图
 */
function loadLifecycleProductHeatmap() {
    fetch('/api/lifecycle_product_heatmap')
        .then(response => response.json())
        .then(data => {
            document.getElementById('lifecycleProductHeatmapChart').classList.remove('loading');
            
            const maxValue = Math.max(...data.data.map(item => item[2]));
            const option = {
                backgroundColor: 'transparent',
                tooltip: {
                    position: 'top',
                    backgroundColor: 'rgba(26, 35, 50, 0.95)',
                    borderColor: '#4facfe',
                    borderWidth: 1,
                    textStyle: { color: '#fff' },
                    boxShadow: '0 0 15px rgba(79, 172, 254, 0.3)'
                },
                animation: true,
                animationDuration: 1000,
                animationEasing: 'cubicOut',
                grid: {
                    height: '50%',
                    y: '10%'
                },
                xAxis: {
                    type: 'category',
                    data: data.xAxis,
                    splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.03)'] } },
                    axisLabel: { color: '#fff' },
                    axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
                    axisTick: { show: false }
                },
                yAxis: {
                    type: 'category',
                    data: data.yAxis,
                    splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.03)'] } },
                    axisLabel: { color: '#fff' },
                    axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
                    axisTick: { show: false }
                },
                visualMap: {
                    min: 0,
                    max: maxValue,
                    calculable: true,
                    orient: 'horizontal',
                    left: 'center',
                    bottom: '15%',
                    textStyle: { color: '#fff' },
                    inRange: {
                        color: ['#1a2332', '#4facfe', '#00f2fe']
                    },
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    itemWidth: 20,
                    itemHeight: 120
                },
                series: [{
                    name: '客户数量',
                    type: 'heatmap',
                    data: data.data,
                    label: {
                        show: true,
                        color: '#fff',
                        fontSize: 11
                    },
                    emphasis: {
                        itemStyle: {
                            shadowBlur: 15,
                            shadowColor: 'rgba(79, 172, 254, 0.8)',
                            borderColor: '#4facfe',
                            borderWidth: 2
                        },
                        label: {
                            color: '#fff',
                            fontWeight: 'bold'
                        }
                    },
                    animationDelay: function(idx) {
                        return idx * 10;
                    }
                }]
            };
            
            lifecycleProductHeatmapChart.setOption(option);
            
            // 实现小方块选中效果
            let selectedItem = null;
            
            // 添加点击事件监听
            lifecycleProductHeatmapChart.on('click', params => {
                // 如果之前有选中的项，清除选中状态
                if (selectedItem !== null) {
                    lifecycleProductHeatmapChart.dispatchAction({
                        type: 'downplay',
                        seriesIndex: 0,
                        dataIndex: selectedItem
                    });
                }
                
                // 选中当前点击的项
                selectedItem = params.dataIndex;
                lifecycleProductHeatmapChart.dispatchAction({
                    type: 'highlight',
                    seriesIndex: 0,
                    dataIndex: selectedItem
                });
                
                // 添加选中项的特殊样式
                lifecycleProductHeatmapChart.dispatchAction({
                    type: 'select',
                    seriesIndex: 0,
                    dataIndex: selectedItem
                });
                
                console.log('选中的项:', params.data);
            });
            
            // 实现自动随机选中小方块的效果
            const totalItems = data.data.length;
            
            // 每1秒自动随机选中一个小方块
            setInterval(() => {
                // 清除之前选中项的状态
                if (selectedItem !== null) {
                    lifecycleProductHeatmapChart.dispatchAction({
                        type: 'downplay',
                        seriesIndex: 0,
                        dataIndex: selectedItem
                    });
                }
                
                // 随机生成一个索引，实现随机选中效果
                const randomIndex = Math.floor(Math.random() * totalItems);
                selectedItem = randomIndex;
                
                // 选中当前随机项
                lifecycleProductHeatmapChart.dispatchAction({
                    type: 'highlight',
                    seriesIndex: 0,
                    dataIndex: selectedItem
                });
                
                // 添加选中项的特殊样式
                lifecycleProductHeatmapChart.dispatchAction({
                    type: 'select',
                    seriesIndex: 0,
                    dataIndex: selectedItem
                });
                
                console.log('自动随机选中项索引:', selectedItem, '数据:', data.data[selectedItem]);
            }, 1000); // 每1000毫秒（1秒）随机切换一次
        })
        .catch(error => console.error('Error fetching lifecycle product heatmap:', error));
}

/**
 * 加载并渲染高潜力客户画像图表
 */
function loadHighPotentialCustomers() {
    fetch('/api/high_potential_customers')
        .then(response => response.json())
        .then(data => {
            document.getElementById('highPotentialCustomersChart').classList.remove('loading');
            
            const option = {
                backgroundColor: 'transparent',
                title: {
                    left: 'center',
                    textStyle: { 
                        color: '#4facfe', 
                        fontSize: 16, 
                        fontWeight: 'normal',
                        textShadow: '0 0 10px rgba(79, 172, 254, 0.5)'
                    }
                },
                tooltip: {
                    trigger: 'axis',
                    axisPointer: { type: 'cross' },
                    backgroundColor: 'rgba(26, 35, 50, 0.95)',
                    borderColor: '#4facfe',
                    borderWidth: 1,
                    textStyle: { color: '#fff' },
                    boxShadow: '0 0 15px rgba(79, 172, 254, 0.3)'
                },
                legend: {
                    data: ['高潜力客户数量', '平均资产(万元)'],
                    textStyle: { color: '#fff' },
                    top: '10%',
                    right: '10%',
                    itemWidth: 10,
                    itemHeight: 10,
                    itemGap: 15
                },
                xAxis: {
                    type: 'category',
                    data: data.ages,
                    axisLabel: { color: '#fff' },
                    axisLine: { lineStyle: { color: 'rgba(79, 172, 254, 0.2)' } },
                    axisTick: { show: false },
                    splitLine: { lineStyle: { color: 'rgba(79, 172, 254, 0.1)' } }
                },
                yAxis: [
                    {
                        type: 'value',
                        name: '客户数量',
                        position: 'left',
                        alignTicks: true,
                        axisLine: { 
                            show: true, 
                            lineStyle: { 
                                color: '#4facfe',
                                shadowBlur: 10,
                                shadowColor: 'rgba(79, 172, 254, 0.7)'
                            } 
                        },
                        axisLabel: { color: '#fff' },
                        axisTick: { show: false },
                        splitLine: { lineStyle: { color: 'rgba(79, 172, 254, 0.1)' } }
                    },
                    {
                        type: 'value',
                        name: '平均资产',
                        position: 'right',
                        alignTicks: true,
                        axisLine: { 
                            show: true, 
                            lineStyle: { 
                                color: '#7b61ff',
                                shadowBlur: 10,
                                shadowColor: 'rgba(123, 97, 255, 0.7)'
                            } 
                        },
                        axisLabel: { color: '#fff' },
                        axisTick: { show: false },
                        splitLine: { show: false }
                    }
                ],
                series: [
                    {
                        name: '高潜力客户数量',
                        type: 'bar',
                        yAxisIndex: 0,
                        itemStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                {offset: 0, color: '#4facfe'},
                                {offset: 1, color: '#00f2fe'}
                            ]),
                            borderRadius: [4, 4, 0, 0],
                            shadowBlur: 15,
                            shadowColor: 'rgba(79, 172, 254, 0.5)',
                            shadowOffsetY: 5
                        },
                        emphasis: {
                            itemStyle: {
                                shadowBlur: 25,
                                shadowColor: 'rgba(79, 172, 254, 0.8)'
                            }
                        },
                        data: data.customer_counts,
                        animationDelay: function(idx) {
                            return idx * 100;
                        }
                    },
                    {
                        name: '平均资产(万元)',
                        type: 'line',
                        yAxisIndex: 1,
                        symbol: 'circle',
                        symbolSize: 8,
                        itemStyle: {
                        color: '#7b61ff',
                        borderColor: '#fff',
                        borderWidth: 2,
                        shadowBlur: 20,
                        shadowColor: 'rgba(123, 97, 255, 0.8)'
                        },
                        lineStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                            {offset: 0, color: '#7b61ff'},
                            {offset: 1, color: '#a389ff'}
                        ]),
                        width: 3,
                        shadowColor: 'rgba(123, 97, 255, 0.5)',
                        shadowBlur: 15
                        },
                        emphasis: {
                        itemStyle: {
                        shadowBlur: 30,
                        shadowColor: 'rgba(123, 97, 255, 1)',
                        symbolSize: 12
                        }
                        },
                        data: data.avg_assets.map(val => Math.round(val) / 10000), // 转换为万元
                        smooth: true
                    },
                    {
                        name: '定位光点',
                        type: 'effectScatter',
                        coordinateSystem: 'cartesian2d',
                        yAxisIndex: 1,
                        showSymbol: false,
                        data: data.avg_assets.map((val, idx) => [idx, Math.round(val) / 10000]),
                        effectType: 'ripple',
                        rippleEffect: {
                        period: 2,
                        scale: 5,
                        brushType: 'fill'
                        },
                        itemStyle: {
                        color: '#7b61ff',
                        shadowBlur: 25,
                        shadowColor: 'rgba(123, 97, 255, 0.8)'
                        },
                        animationDelay: function (idx) {
                        return idx * 300;
                        },
                        animationEasing: 'cubicOut',
                        animationDelayUpdate: function (idx) {
                        return idx * 300;
                    }
                    }
                ],
                animationEasing: 'elasticOut',
                animationDelayUpdate: function(idx) {
                    return idx * 5;
                },
                animationDuration: 1500
            };
            
            highPotentialCustomersChart.setOption(option);
        })
        .catch(error => console.error('Error fetching high potential customers:', error));
}

/**
 * 加载并渲染营销效果监控图表
 */
function loadMarketingPerformance() {
    fetch('/api/marketing_performance')
        .then(response => response.json())
        .then(data => {
            document.getElementById('marketingPerformanceChart').classList.remove('loading');
            const successRate = Math.round(data.success_rate * 100);
            document.getElementById('marketing-success').textContent = `${successRate}%`;
            animateProgress('marketing-success', successRate);
            
            // 更新HTML标题，将数据融入其中
            document.getElementById('marketing-title').textContent = 
                `营销触达效果监控 - 成功率: ${(data.success_rate * 100).toFixed(2)}%`;
            
            const option = {
                backgroundColor: 'transparent',
                tooltip: {
                    trigger: 'item',
                    formatter: '{a} <br/>{b}: {c} ({d}%)',
                    backgroundColor: 'rgba(26, 35, 50, 0.95)',
                    borderColor: '#4facfe',
                    borderWidth: 1,
                    textStyle: { color: '#fff' },
                    boxShadow: '0 0 15px rgba(79, 172, 254, 0.3)'
                },
                series: [
                    {
                        name: '触达结果',
                        type: 'pie',
                        radius: '60%',
                        center: ['50%', '60%'],
                        data: data.results.map((result, index) => ({
                            value: data.counts[index],
                            name: result
                        })),
                        emphasis: {
                            itemStyle: {
                                shadowBlur: 20,
                                shadowOffsetX: 0,
                                shadowColor: 'rgba(79, 172, 254, 0.6)',
                                scale: 1.05
                            }
                        },
                        label: {
                            color: '#fff',
                            fontSize: 12
                        },
                        itemStyle: {
                            borderRadius: 10,
                            borderColor: '#1a2332',
                            borderWidth: 2,
                            color: function(params) {
                                const colorList = [
                                    new echarts.graphic.LinearGradient(0, 0, 0, 1, [{offset: 0, color: '#4facfe'}, {offset: 1, color: '#3a90d6'}]),
                                    new echarts.graphic.LinearGradient(0, 0, 0, 1, [{offset: 0, color: '#00f2fe'}, {offset: 1, color: '#00d4d4'}]),
                                    new echarts.graphic.LinearGradient(0, 0, 0, 1, [{offset: 0, color: '#ff6b6b'}, {offset: 1, color: '#ff5252'}]),
                                    new echarts.graphic.LinearGradient(0, 0, 0, 1, [{offset: 0, color: '#ffd166'}, {offset: 1, color: '#ffc133'}])
                                ];
                                return colorList[params.dataIndex];
                            },
                            shadowBlur: 15,
                            shadowColor: 'rgba(0, 0, 0, 0.3)'
                        },
                        animationType: 'scale',
                        animationEasing: 'elasticOut',
                        animationDelay: function(idx) {
                            return Math.random() * 200;
                        },
                        animationDuration: 1500
                    }
                ]
            };
            
            marketingPerformanceChart.setOption(option);
        })
        .catch(error => console.error('Error fetching marketing performance:', error));
}

/**
 * 加载并渲染地域分布图表
 */
function loadGeographicDistribution() {
    fetch('/api/geographic_distribution')
        .then(response => response.json())
        .then(data => {
            document.getElementById('geographicDistributionChart').classList.remove('loading');
            
            const option = {
                backgroundColor: 'transparent',
                title: {
                    left: 'center',
                    textStyle: { 
                        color: '#4facfe', 
                        fontSize: 16, 
                        fontWeight: 'normal',
                        textShadow: '0 0 10px rgba(79, 172, 254, 0.5)'
                    }
                },
                tooltip: {
                    trigger: 'axis',
                    axisPointer: { type: 'cross' },
                    backgroundColor: 'rgba(26, 35, 50, 0.95)',
                    borderColor: '#4facfe',
                    borderWidth: 1,
                    textStyle: { color: '#fff' },
                    boxShadow: '0 0 15px rgba(79, 172, 254, 0.3)'
                },
                legend: {
                    data: ['客户数量', '平均资产(万元)'],
                    textStyle: { color: '#fff' },
                    top: '10%',
                    right: '10%',
                    itemWidth: 10,
                    itemHeight: 10,
                    itemGap: 15
                },
                xAxis: {
                    type: 'category',
                    data: data.city_levels,
                    axisLabel: { color: '#fff' },
                    axisLine: { lineStyle: { color: 'rgba(79, 172, 254, 0.2)' } },
                    axisTick: { show: false },
                    splitLine: { lineStyle: { color: 'rgba(79, 172, 254, 0.1)' } }
                },
                yAxis: [
                    {
                        type: 'value',
                        name: '客户数量',
                        position: 'left',
                        axisLine: { 
                            show: true, 
                            lineStyle: { 
                                color: '#4facfe',
                                shadowBlur: 10,
                                shadowColor: 'rgba(79, 172, 254, 0.7)'
                            } 
                        },
                        axisLabel: { color: '#fff' },
                        axisTick: { show: false },
                        splitLine: { lineStyle: { color: 'rgba(79, 172, 254, 0.1)' } }
                    },
                    {
                        type: 'value',
                        name: '平均资产(万元)',
                        position: 'right',
                        axisLine: { 
                            show: true, 
                            lineStyle: { 
                                color: '#7b61ff',
                                shadowBlur: 10,
                                shadowColor: 'rgba(123, 97, 255, 0.7)'
                            } 
                        },
                        axisLabel: { color: '#fff' },
                        axisTick: { show: false },
                        splitLine: { show: false }
                    }
                ],
                series: [
                    {
                        name: '客户数量',
                        type: 'bar',
                        yAxisIndex: 0,
                        itemStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                {offset: 0, color: '#4facfe'},
                                {offset: 1, color: '#00f2fe'}
                            ]),
                            borderRadius: [4, 4, 0, 0],
                            shadowBlur: 15,
                            shadowColor: 'rgba(79, 172, 254, 0.5)',
                            shadowOffsetY: 5
                        },
                        emphasis: {
                            itemStyle: {
                                shadowBlur: 25,
                                shadowColor: 'rgba(79, 172, 254, 0.8)'
                            }
                        },
                        data: data.customer_counts,
                        animationDelay: function(idx) {
                            return idx * 100;
                        }
                    },
                    {
                        name: '平均资产(万元)',
                        type: 'line',
                        yAxisIndex: 1,
                        symbol: 'circle',
                        symbolSize: 8,
                        itemStyle: {
                            color: '#7b61ff',
                            borderColor: '#fff',
                            borderWidth: 2,
                            shadowBlur: 20,
                            shadowColor: 'rgba(123, 97, 255, 0.8)'
                        },
                        lineStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                                {offset: 0, color: '#7b61ff'},
                                {offset: 1, color: '#a389ff'}
                            ]),
                            width: 3,
                            shadowColor: 'rgba(123, 97, 255, 0.5)',
                            shadowBlur: 15
                        },
                        emphasis: {
                            itemStyle: {
                                shadowBlur: 30,
                                shadowColor: 'rgba(123, 97, 255, 1)',
                                symbolSize: 12
                            }
                        },
                        data: data.avg_assets.map(val => Math.round(val) / 10000), // 转换为万元
                        smooth: true
                    },
                    {
                        name: '定位光点',
                        type: 'effectScatter',
                        coordinateSystem: 'cartesian2d',
                        yAxisIndex: 1,
                        showSymbol: false,
                        data: data.avg_assets.map((val, idx) => [idx, Math.round(val) / 10000]),
                        effectType: 'ripple',
                        rippleEffect: {
                            period: 2,
                            scale: 5,
                            brushType: 'fill'
                        },
                        itemStyle: {
                            color: '#7b61ff',
                            shadowBlur: 25,
                            shadowColor: 'rgba(123, 97, 255, 0.8)'
                        },
                        animationDelay: function (idx) {
                            return idx * 300;
                        },
                        animationEasing: 'cubicOut',
                        animationDelayUpdate: function (idx) {
                            return idx * 300;
                        }
                    }
                ],
                animationEasing: 'elasticOut',
                animationDelayUpdate: function(idx) {
                    return idx * 5;
                },
                animationDuration: 1500
            };
            
            geographicDistributionChart.setOption(option);
        })
        .catch(error => console.error('Error fetching geographic distribution:', error));
}

/**
 * 加载仪表板摘要数据
 */
function loadDashboardSummary() {
    fetch('/api/dashboard_summary')
        .then(response => response.json())
        .then(summary => {
            // 更新总客户数
            animateValue('total-customers', 0, summary.total_customers, 2000);
            
            // 更新平均资产
            animateValue('avg-assets', 0, Math.round(summary.avg_assets / 10000), 2000); // 转换为万元
            
            // 更新活跃客户占比
            animateValue('active-customers', 0, Math.round(summary.active_rate), 2000);
            animateProgress('active-customers', Math.round(summary.active_rate));
        })
        .catch(error => console.error('Error fetching dashboard summary:', error));
}

/**
 * 初始化整个仪表板
 */
function initDashboard() {
    // 初始化图表
    initCharts();
    
    // 加载各类数据
    loadDashboardSummary();
    loadAssetDistribution();
    loadLifecycleProductHeatmap();
    loadHighPotentialCustomers();
    loadMarketingPerformance();
    loadGeographicDistribution();
}

// 页面加载完成后初始化仪表板
document.addEventListener('DOMContentLoaded', initDashboard);

// 导出公共方法，便于其他脚本调用
window.Dashboard = {
    init: initDashboard,
    refresh: function() {
        loadDashboardSummary();
        loadAssetDistribution();
        loadLifecycleProductHeatmap();
        loadHighPotentialCustomers();
        loadMarketingPerformance();
        loadGeographicDistribution();
    }
};