def generated_function_dkdxdlogsigma(logsigma, logtheta, x, y):
    return -2*ehoch1**logtheta*(ehoch1**logsigma*x - ehoch1**logsigma*y)/(pi*np.sqrt(-(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) + 1)*(ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)**(3/2)*np.sqrt(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) + ehoch1**logtheta*(ehoch1**logsigma*x - ehoch1**logsigma*y - y)*(ehoch1**logsigma*(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)**2) + ehoch1**logsigma*(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)**2*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) - 2*ehoch1**logsigma*(ehoch1**logsigma + ehoch1**logtheta*x*y)/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)))/(pi*(-(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) + 1)**(3/2)*(ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)**(3/2)*np.sqrt(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) + ehoch1**(logsigma + logtheta)*(ehoch1**logsigma*x - ehoch1**logsigma*y - y)/(pi*np.sqrt(-(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) + 1)*(ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)**(3/2)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)**(3/2)) + 3*ehoch1**(logsigma + logtheta)*(ehoch1**logsigma*x - ehoch1**logsigma*y - y)/(pi*np.sqrt(-(ehoch1**logsigma + ehoch1**logtheta*x*y)**2/((ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)*(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1)) + 1)*(ehoch1**logsigma + ehoch1**logtheta*x**2 + 1)**(5/2)*np.sqrt(ehoch1**logsigma + ehoch1**logtheta*y**2 + 1))
