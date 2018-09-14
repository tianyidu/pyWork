package com.wiwj.www.test;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.apache.http.HttpEntity;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.FileEntity;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;

/**
 * Hello world!
 *
 */
public class httpclientPost 
{
    public static void main( String[] args ) throws IOException
    {
    	CloseableHttpClient httpclient = HttpClients.createDefault();
    	//添加图片
    	MultipartEntityBuilder builder = MultipartEntityBuilder.create();
    	File file = new File("E:\\PWORKSPACE\\house3\\pic3\\bedroom\\1cb71518-d7a9-4bab-8acb-e8cef1dba747.jpg");
    	builder.addBinaryBody("img_test", file);
    	HttpEntity entity = builder.build();
    	
    	//请求接口
    	HttpPost httppost = new HttpPost("http://127.0.0.1:8000/cnn/api");
    	httppost.setEntity(entity);

    	//获取输出
    	CloseableHttpResponse response = null;
    	try {
    		response = httpclient.execute(httppost);
    		System.out.println(response);
    		BufferedReader br = new BufferedReader(new InputStreamReader(response.getEntity().getContent()));
    		
    		System.out.println(br.readLine());
		} catch (ClientProtocolException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {
			response.close();
		}
    	
    	
    }
}
